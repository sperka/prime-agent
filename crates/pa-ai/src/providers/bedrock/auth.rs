//! Bedrock authentication and endpoint resolution: the `SigV4` request signing used by
//! `@aws-sdk/client-bedrock-runtime` for `POST /model/{modelId}/converse-stream`, plus the region /
//! endpoint / credential resolution rules from the TS provider.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::io::ErrorKind;
use std::path::Path;
use std::process::Stdio;
use std::sync::{Mutex, PoisonError};

use hmac::{Hmac, Mac};
use sha2::{Digest, Sha256};

use crate::providers::bedrock::BedrockOptions;
use crate::types::Model;

type HmacSha256 = Hmac<Sha256>;

/// Resolved AWS credentials (env vars, profile config, or SigV4-skip dummy).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AwsCredentials {
    pub access_key_id: String,
    pub secret_access_key: String,
    pub session_token: Option<String>,
}

fn env(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|value| !value.is_empty())
}

/// Parse an AWS credentials-style ini file: `[section-name]` headers with `key = value` pairs (only
/// the two credential lines are needed).
fn parse_ini_credentials(text: &str, section: &str) -> Option<AwsCredentials> {
    let mut in_section = false;
    let mut access_key_id = None;
    let mut secret_access_key = None;
    let mut session_token = None;
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with('[') && trimmed.ends_with(']') {
            let name = trimmed[1..trimmed.len() - 1].trim();
            // AWS credential files use `[profile name]` (config) or `[name]`.
            in_section = name == section
                || name == format!("profile {section}")
                || name.split_whitespace().last() == Some(section);
            continue;
        }
        if !in_section {
            continue;
        }
        let Some((key, value)) = trimmed.split_once('=') else {
            continue;
        };
        let value = value.trim().to_string();
        match key.trim() {
            "aws_access_key_id" => access_key_id = Some(value),
            "aws_secret_access_key" => secret_access_key = Some(value),
            "aws_session_token" => session_token = Some(value),
            _ => {}
        }
    }
    Some(AwsCredentials {
        access_key_id: access_key_id?,
        secret_access_key: secret_access_key?,
        session_token,
    })
}

/// The error text when no source yields credentials. A failing `credential_process` appends its
/// cause after this prefix.
const NO_CREDENTIALS: &str = "No AWS credentials available for Bedrock";

/// Seconds before the expiry at which cached `credential_process` credentials count as stale. The
/// AWS SDK refreshes memoized credentials five minutes before they expire.
const PROCESS_CREDENTIALS_REFRESH_SECS: i64 = 300;

/// `credential_process` credentials that carry an `Expiration`, keyed by command string, with the
/// expiry in Unix seconds. The TS provider builds a new SDK client per request, so its
/// `credential_process` runs on every request, and a process such as `isengardcli` takes about half
/// a second. This cache serves a result until `PROCESS_CREDENTIALS_REFRESH_SECS` before it expires.
/// Credentials without an `Expiration` are never cached, so each request runs the process again.
static PROCESS_CREDENTIALS: Mutex<BTreeMap<String, (AwsCredentials, i64)>> =
    Mutex::new(BTreeMap::new());

/// Version 1 JSON that a `credential_process` prints. It has no `Debug` so that credential values
/// cannot reach a log or an error.
#[derive(serde::Deserialize)]
struct ProcessOutput {
    #[serde(rename = "Version")]
    version: Option<u64>,
    #[serde(rename = "AccessKeyId")]
    access_key_id: Option<String>,
    #[serde(rename = "SecretAccessKey")]
    secret_access_key: Option<String>,
    #[serde(rename = "SessionToken")]
    session_token: Option<String>,
    #[serde(rename = "Expiration")]
    expiration: Option<String>,
}

/// The SDK credential chain used by the TS provider: SigV4-skip dummy keys, static env credentials,
/// the shared credentials file (default or `AWS_PROFILE` / `options.profile`), then the
/// `credential_process` of that profile in the shared config file.
///
/// The error text is `No AWS credentials available for Bedrock`, with the cause of a failing
/// `credential_process` or an unreadable file appended.
pub(crate) async fn resolve_credentials(profile: Option<&str>) -> Result<AwsCredentials, String> {
    if std::env::var("AWS_BEDROCK_SKIP_AUTH").as_deref() == Ok("1") {
        return Ok(AwsCredentials {
            access_key_id: "dummy-access-key".to_string(),
            secret_access_key: "dummy-secret-key".to_string(),
            session_token: None,
        });
    }
    if let (Some(access_key_id), Some(secret_access_key)) =
        (env("AWS_ACCESS_KEY_ID"), env("AWS_SECRET_ACCESS_KEY"))
    {
        return Ok(AwsCredentials {
            access_key_id,
            secret_access_key,
            session_token: env("AWS_SESSION_TOKEN"),
        });
    }
    let profile = profile
        .map(std::string::ToString::to_string)
        .or_else(|| env("AWS_PROFILE"))
        .unwrap_or_else(|| "default".to_string());
    let Some(home) = pa_types::platform::home_dir() else {
        return Err(NO_CREDENTIALS.to_string());
    };
    profile_credentials(&home.join(".aws"), &profile).await
}

/// The static keys of `profile` in `aws_dir/credentials`, else the `credential_process` of
/// `profile` in `aws_dir/config`. A file that does not exist counts as empty. A credentials file
/// that cannot be read is the error only when the config file yields no credentials either.
async fn profile_credentials(aws_dir: &Path, profile: &str) -> Result<AwsCredentials, String> {
    let read = |name: &str| {
        let path = aws_dir.join(name);
        match std::fs::read_to_string(&path) {
            Ok(text) => Ok(text),
            Err(error) if error.kind() == ErrorKind::NotFound => Ok(String::new()),
            Err(error) => Err(format!(
                "{NO_CREDENTIALS}: could not read {}: {error}",
                path.display()
            )),
        }
    };
    let credentials_text = read("credentials");
    if let Some(credentials) = credentials_text
        .as_deref()
        .ok()
        .and_then(|text| parse_ini_credentials(text, profile))
    {
        return Ok(credentials);
    }
    if let Some(credentials) = credentials_from_process(profile, &read("config")?).await? {
        return Ok(credentials);
    }
    Err(credentials_text
        .err()
        .unwrap_or_else(|| NO_CREDENTIALS.to_string()))
}

/// Run the `credential_process` of `profile` found in `~/.aws/config` text and parse its version 1
/// JSON. Returns `Ok(None)` when the profile has no `credential_process`.
async fn credentials_from_process(
    profile: &str,
    config_text: &str,
) -> Result<Option<AwsCredentials>, String> {
    let Some(command) = config_value(config_text, profile, "credential_process")
        .filter(|command| !command.is_empty())
    else {
        return Ok(None);
    };
    let failure = |detail: &str| {
        format!("{NO_CREDENTIALS}: credential_process for profile '{profile}' {detail}")
    };
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |duration| {
            i64::try_from(duration.as_secs()).unwrap_or(i64::MAX)
        });

    // The guard is a temporary of this statement, so it is gone before the process runs.
    let cached = PROCESS_CREDENTIALS
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .get(&command)
        .filter(|(_, expiry)| expiry - now > PROCESS_CREDENTIALS_REFRESH_SECS)
        .map(|(credentials, _)| credentials.clone());
    if let Some(credentials) = cached {
        return Ok(Some(credentials));
    }

    let (shell, flag) = if cfg!(windows) {
        ("cmd", "/C")
    } else {
        ("sh", "-c")
    };
    // Both output streams are piped: the TUI owns the terminal, so stderr must not reach it.
    let output = tokio::process::Command::new(shell)
        .arg(flag)
        .arg(&command)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true)
        .output()
        .await
        .map_err(|error| failure(&format!("could not start: {error}")))?;
    if !output.status.success() {
        let status = output.status;
        let stderr = String::from_utf8_lossy(&output.stderr);
        let detail = match stderr.lines().map(str::trim).find(|line| !line.is_empty()) {
            Some(line) => {
                let line = line.chars().take(200).collect::<String>();
                format!("exited with {status}: {line}")
            }
            None => format!("exited with {status}"),
        };
        return Err(failure(&detail));
    }

    // The causes and their wording follow the JS SDK. The serde_json error is not passed on
    // because its text can quote a credential value.
    let Ok(parsed) = serde_json::from_slice::<ProcessOutput>(&output.stdout) else {
        return Err(failure("returned invalid JSON"));
    };
    if parsed.version != Some(1) {
        return Err(failure("did not return Version 1"));
    }
    let (Some(access_key_id), Some(secret_access_key)) = (
        parsed.access_key_id.filter(|key| !key.is_empty()),
        parsed.secret_access_key.filter(|key| !key.is_empty()),
    ) else {
        return Err(failure("returned invalid credentials"));
    };
    // An `Expiration` that does not parse counts as none, as in the JS SDK.
    let expiry = parsed.expiration.as_deref().and_then(unix_from_rfc3339);
    if expiry.is_some_and(|expiry| expiry <= now) {
        return Err(failure("returned expired credentials"));
    }
    let credentials = AwsCredentials {
        access_key_id,
        secret_access_key,
        session_token: parsed.session_token.filter(|token| !token.is_empty()),
    };
    if let Some(expiry) = expiry {
        PROCESS_CREDENTIALS
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .insert(command, (credentials.clone(), expiry));
    }
    Ok(Some(credentials))
}

/// Parse an RFC 3339 timestamp such as `2026-10-23T16:13:07.5+02:00` into Unix seconds. Fractional
/// seconds are ignored and the offset is required. Returns `None` for any other shape.
fn unix_from_rfc3339(text: &str) -> Option<i64> {
    let number = |digits: &[u8]| -> Option<i64> {
        digits.iter().all(u8::is_ascii_digit).then(|| {
            digits
                .iter()
                .fold(0, |total, digit| total * 10 + i64::from(digit - b'0'))
        })
    };
    let bytes = text.as_bytes();
    let separators = [(4, b'-'), (7, b'-'), (10, b'T'), (13, b':'), (16, b':')];
    if !separators
        .iter()
        .all(|&(index, separator)| bytes.get(index) == Some(&separator))
    {
        return None;
    }
    let year = number(bytes.get(0..4)?)?;
    let month = number(bytes.get(5..7)?)?;
    let day = number(bytes.get(8..10)?)?;
    let hour = number(bytes.get(11..13)?)?;
    let minute = number(bytes.get(14..16)?)?;
    let second = number(bytes.get(17..19)?)?;
    if !(1..=12).contains(&month)
        || !(1..=31).contains(&day)
        || hour > 23
        || minute > 59
        || second > 60
    {
        return None;
    }

    let mut rest = text.get(19..)?;
    if let Some(fraction) = rest.strip_prefix('.') {
        let fraction_digits = fraction.bytes().take_while(u8::is_ascii_digit).count();
        if fraction_digits == 0 {
            return None;
        }
        rest = &fraction[fraction_digits..];
    }
    let offset_seconds = if rest == "Z" {
        0
    } else {
        let offset = rest.as_bytes();
        let sign = match offset.first()? {
            b'+' => 1,
            b'-' => -1,
            _ => return None,
        };
        if offset.len() != 6 || offset[3] != b':' {
            return None;
        }
        let (offset_hour, offset_minute) = (number(&offset[1..3])?, number(&offset[4..6])?);
        if offset_hour > 23 || offset_minute > 59 {
            return None;
        }
        sign * (offset_hour * 3600 + offset_minute * 60)
    };

    // Days since 1970-01-01 (Howard Hinnant's days_from_civil, the inverse of `civil_from_unix`).
    let shifted_year = if month <= 2 { year - 1 } else { year };
    let era = shifted_year.div_euclid(400);
    let year_of_era = shifted_year - era * 400;
    let day_of_year = (153 * (if month > 2 { month - 3 } else { month + 9 }) + 2) / 5 + day - 1;
    let day_of_era = year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year;
    let days = era * 146_097 + day_of_era - 719_468;
    Some(days * 86_400 + hour * 3600 + minute * 60 + second - offset_seconds)
}

fn sha256_hex(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

fn hmac_sha256(key: &[u8], data: &[u8]) -> Vec<u8> {
    let mut mac = HmacSha256::new_from_slice(key).expect("hmac accepts any key length");
    mac.update(data);
    mac.finalize().into_bytes().to_vec()
}

/// Convert a Unix timestamp to civil UTC date-time (Howard Hinnant's algorithm).
fn civil_from_unix(secs: u64) -> (i64, u32, u32, u32, u32, u32) {
    // u64::MAX / 86_400 is ~2.1e14; the day count always fits i64.
    let days = i64::try_from(secs / 86_400).expect("days since the epoch fit i64");
    let secs_of_day = secs % 86_400;
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    // Howard Hinnant's algorithm: day 1..=31, month 1..=12 for any input day count.
    let d = u32::try_from(doy - (153 * mp + 2) / 5 + 1).expect("day of month is 1..=31");
    let m = u32::try_from(if mp < 10 { mp + 3 } else { mp - 9 }).expect("month is 1..=12");
    let year = if m <= 2 { y + 1 } else { y };
    (
        year,
        m,
        d,
        (secs_of_day / 3600) as u32,
        ((secs_of_day % 3600) / 60) as u32,
        (secs_of_day % 60) as u32,
    )
}

/// Inputs of one `SigV4` signature.
pub struct SigV4Params<'a> {
    pub method: &'a str,
    /// Request path, already percent-encoded once, with no query string. The canonical query line
    /// is therefore always empty.
    pub path: &'a str,
    pub host: &'a str,
    pub region: &'a str,
    pub service: &'a str,
    pub body: &'a [u8],
    /// Extra headers included in the signed header list (lowercase names).
    pub extra_signed_headers: &'a [(String, String)],
    /// The signing time.
    pub timestamp: std::time::SystemTime,
}

/// Produce the `SigV4` auth headers for a request. Returns `(x-amz-date, authorization,
/// x-amz-security-token)`.
pub fn sigv4_headers(
    params: &SigV4Params<'_>,
    credentials: &AwsCredentials,
) -> (String, String, Option<String>) {
    // `SigV4` timestamp formats: `x-amz-date` (20250101T000000Z) and date stamp (20250101). Uses std
    // time to avoid a chrono dependency.
    let seconds = params
        .timestamp
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |duration| duration.as_secs());
    let (year, month, day, hour, minute, second) = civil_from_unix(seconds);
    let amz_date = format!("{year:04}{month:02}{day:02}T{hour:02}{minute:02}{second:02}Z");
    let date_stamp = format!("{year:04}{month:02}{day:02}");
    let payload_hash = sha256_hex(params.body);

    let mut canonical_headers: Vec<(String, String)> = vec![
        ("content-type".to_string(), "application/json".to_string()),
        ("host".to_string(), params.host.to_string()),
        ("x-amz-date".to_string(), amz_date.clone()),
    ];
    for (name, value) in params.extra_signed_headers {
        canonical_headers.push((name.to_lowercase(), value.clone()));
    }
    if let Some(token) = &credentials.session_token {
        canonical_headers.push(("x-amz-security-token".to_string(), token.clone()));
    }
    canonical_headers.sort_by(|a, b| a.0.cmp(&b.0));

    let signed_headers = canonical_headers
        .iter()
        .map(|(name, _)| name.as_str())
        .collect::<Vec<_>>()
        .join(";");
    let canonical_headers_text =
        canonical_headers
            .iter()
            .fold(String::new(), |mut text, (name, value)| {
                let _ = writeln!(text, "{name}:{value}");
                text
            });

    // Every service except S3 double-encodes the path in the canonical URI, which is what the AWS
    // SDK signer does: the escaped model id `v1%3A0` is signed as `v1%253A0`. Only the signed copy
    // changes, the request URL keeps the single encoding.
    let mut canonical_uri = String::new();
    for byte in params.path.bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b'~' | b'/') {
            canonical_uri.push(char::from(byte));
        } else {
            let _ = write!(canonical_uri, "%{byte:02X}");
        }
    }

    // The empty line after the URI is the canonical query string, which is always empty here.
    let method = params.method;
    let canonical_request = format!(
        "{method}\n{canonical_uri}\n\n{canonical_headers_text}\n{signed_headers}\n{payload_hash}"
    );

    let credential_scope = format!(
        "{}/{}/{}/aws4_request",
        date_stamp, params.region, params.service
    );
    let string_to_sign = format!(
        "AWS4-HMAC-SHA256\n{amz_date}\n{credential_scope}\n{}",
        sha256_hex(canonical_request.as_bytes())
    );

    let date_key = hmac_sha256(
        format!("AWS4{}", credentials.secret_access_key).as_bytes(),
        date_stamp.as_bytes(),
    );
    let region_key = hmac_sha256(&date_key, params.region.as_bytes());
    let service_key = hmac_sha256(&region_key, params.service.as_bytes());
    let signing_key = hmac_sha256(&service_key, b"aws4_request");
    let signature = hex::encode(hmac_sha256(&signing_key, string_to_sign.as_bytes()));

    let authorization = format!(
        "AWS4-HMAC-SHA256 Credential={}/{}, SignedHeaders={}, Signature={}",
        credentials.access_key_id, credential_scope, signed_headers, signature
    );

    (amz_date, authorization, credentials.session_token.clone())
}

pub fn get_configured_bedrock_region(region: Option<&str>) -> Option<String> {
    region
        .map(std::string::ToString::to_string)
        .or_else(|| env("AWS_REGION"))
        .or_else(|| env("AWS_DEFAULT_REGION"))
}

pub fn has_configured_bedrock_profile() -> bool {
    env("AWS_PROFILE").is_some()
}

pub fn get_standard_bedrock_endpoint_region(base_url: &str) -> Option<String> {
    let hostname = url::Url::parse(base_url).ok()?.host_str()?.to_lowercase();
    let suffix_stripped = hostname
        .strip_prefix("bedrock-runtime-fips")
        .or_else(|| hostname.strip_prefix("bedrock-runtime"))?;
    let rest = suffix_stripped.strip_prefix('.')?;
    let region = rest
        .strip_suffix(".amazonaws.com")
        .or_else(|| rest.strip_suffix(".amazonaws.com.cn"))?;
    if region.is_empty()
        || !region
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-')
    {
        return None;
    }
    Some(region.to_string())
}

pub fn should_use_explicit_bedrock_endpoint(
    base_url: &str,
    configured_region: Option<&str>,
    has_configured_profile: bool,
) -> bool {
    let endpoint_region = get_standard_bedrock_endpoint_region(base_url);
    match endpoint_region {
        None => true,
        Some(_) => configured_region.is_none() && !has_configured_profile,
    }
}

/// Resolve the request endpoint and region: explicit model baseUrl (custom gateways, fips,
/// `GovCloud`) or the standard regional endpoint.
pub(crate) fn resolve_endpoint(model: &Model, options: &BedrockOptions) -> (String, String) {
    let configured_region = get_configured_bedrock_region(options.region.as_deref());
    let has_profile = has_configured_bedrock_profile();
    let use_explicit_endpoint = should_use_explicit_bedrock_endpoint(
        &model.base_url,
        configured_region.as_deref(),
        has_profile,
    );

    if use_explicit_endpoint && !model.base_url.is_empty() {
        // Region resolution mirrors the TS: explicit option > env vars > endpoint hostname >
        // (profile-resolved) > us-east-1.
        let region = configured_region
            .or_else(|| {
                crate::providers::bedrock::auth::get_standard_bedrock_endpoint_region(
                    &model.base_url,
                )
            })
            .or_else(env_region_or_profile)
            .unwrap_or_else(|| "us-east-1".to_string());
        return (model.base_url.trim_end_matches('/').to_string(), region);
    }

    let region = configured_region
        .or_else(env_region_or_profile)
        .unwrap_or_else(|| "us-east-1".to_string());
    (
        format!("https://bedrock-runtime.{region}.amazonaws.com"),
        region,
    )
}

/// AWS_PROFILE-resolved region from `~/.aws/config`, else None.
fn env_region_or_profile() -> Option<String> {
    if let Some(region) = std::env::var("AWS_REGION").ok().filter(|v| !v.is_empty()) {
        return Some(region);
    }
    if let Some(region) = std::env::var("AWS_DEFAULT_REGION")
        .ok()
        .filter(|v| !v.is_empty())
    {
        return Some(region);
    }
    let profile = std::env::var("AWS_PROFILE")
        .ok()
        .filter(|v| !v.is_empty())?;
    let home = pa_types::platform::home_dir()?;
    let text = std::fs::read_to_string(home.join(".aws/config")).ok()?;
    config_value(&text, &profile, "region")
}

/// The value of `key` in the `[profile {profile}]` (or `[{profile}]`) section of `~/.aws/config`
/// text, else None.
fn config_value(config_text: &str, profile: &str, key: &str) -> Option<String> {
    let mut in_section = false;
    for line in config_text.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with('[') && trimmed.ends_with(']') {
            let name = trimmed[1..trimmed.len() - 1].trim();
            in_section = name == format!("profile {profile}") || name == profile;
            continue;
        }
        if !in_section {
            continue;
        }
        if let Some((candidate, value)) = trimmed.split_once('=') {
            if candidate.trim() == key {
                return Some(value.trim().to_string());
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, UNIX_EPOCH};

    use super::{
        config_value, credentials_from_process, profile_credentials, sigv4_headers,
        unix_from_rfc3339, AwsCredentials, SigV4Params,
    };

    // The expected values come from botocore 1.42.97 `SigV4Auth(creds, "bedrock",
    // "us-east-1").add_auth(...)` with the documented AWS example keys, signing the same request at
    // the same time (2026-10-23T08:45:17Z).

    #[test]
    fn sigv4_headers_match_botocore_for_a_session_token_request() {
        let credentials = AwsCredentials {
            access_key_id: "AKIDEXAMPLE".to_string(),
            secret_access_key: "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY".to_string(),
            session_token: Some("session-token-example".to_string()),
        };
        let signed = sigv4_headers(
            &SigV4Params {
                method: "POST",
                path: "/model/global.anthropic.claude-haiku-5-5/converse-stream",
                host: "bedrock-runtime.us-east-1.amazonaws.com",
                region: "us-east-1",
                service: "bedrock",
                body: br#"{"messages":[]}"#,
                extra_signed_headers: &[],
                timestamp: UNIX_EPOCH + Duration::from_secs(1_792_745_117),
            },
            &credentials,
        );

        assert_eq!(
            signed,
            (
                "20261023T084517Z".to_string(),
                "AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/20261023/us-east-1/bedrock/aws4_request, \
                 SignedHeaders=content-type;host;x-amz-date;x-amz-security-token, \
                 Signature=33bddde0f56441c5ccf0701c613870b3f41fc5ca973c7902ea97a3c0844642a7"
                    .to_string(),
                Some("session-token-example".to_string()),
            )
        );
    }

    #[test]
    fn sigv4_headers_double_encode_an_escaped_model_id_like_botocore() {
        let credentials = AwsCredentials {
            access_key_id: "AKIDEXAMPLE".to_string(),
            secret_access_key: "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY".to_string(),
            session_token: None,
        };
        let signed = sigv4_headers(
            &SigV4Params {
                method: "POST",
                path: "/model/anthropic.claude-opus-4-5-20251101-v1%3A0/converse-stream",
                host: "bedrock-runtime.us-east-1.amazonaws.com",
                region: "us-east-1",
                service: "bedrock",
                body: br#"{"messages":[]}"#,
                extra_signed_headers: &[(
                    "x-amzn-bedrock-request-metadata".to_string(),
                    r#"{"team":"a"}"#.to_string(),
                )],
                timestamp: UNIX_EPOCH + Duration::from_secs(1_792_745_117),
            },
            &credentials,
        );

        assert_eq!(
            signed,
            (
                "20261023T084517Z".to_string(),
                "AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/20261023/us-east-1/bedrock/aws4_request, \
                 SignedHeaders=content-type;host;x-amz-date;x-amzn-bedrock-request-metadata, \
                 Signature=62eec0df0a8ee67f9742219cb19bd8662413fc550b227f6a4dc8187ddcc06225"
                    .to_string(),
                None,
            )
        );
    }

    const EXAMPLE_CONFIG: &str = "[default]
credential_process = ada credentials print --account 123
region = us-west-2

[profile a]
credential_process = isengardcli credentials --region=us-east-1
region = eu-west-1

[profile b]
region = ap-south-1
";

    #[test]
    fn config_value_returns_the_credential_process_of_the_requested_profile() {
        assert_eq!(
            [
                config_value(EXAMPLE_CONFIG, "a", "credential_process"),
                config_value(EXAMPLE_CONFIG, "default", "credential_process"),
                config_value(EXAMPLE_CONFIG, "b", "credential_process"),
                config_value(EXAMPLE_CONFIG, "missing", "credential_process"),
            ],
            [
                Some("isengardcli credentials --region=us-east-1".to_string()),
                Some("ada credentials print --account 123".to_string()),
                None,
                None,
            ]
        );
    }

    #[test]
    fn config_value_returns_the_region_of_the_requested_profile() {
        assert_eq!(
            [
                config_value(EXAMPLE_CONFIG, "a", "region"),
                config_value(EXAMPLE_CONFIG, "default", "region"),
                config_value(EXAMPLE_CONFIG, "b", "region"),
                config_value(EXAMPLE_CONFIG, "missing", "region"),
            ],
            [
                Some("eu-west-1".to_string()),
                Some("us-west-2".to_string()),
                Some("ap-south-1".to_string()),
                None,
            ]
        );
    }

    #[test]
    fn rfc3339_timestamps_parse_to_unix_seconds() {
        assert_eq!(
            [
                unix_from_rfc3339("1970-01-01T00:00:00Z"),
                unix_from_rfc3339("2000-03-01T00:00:00.123Z"),
                unix_from_rfc3339("2026-10-23T16:13:07Z"),
                unix_from_rfc3339("2026-10-23T18:13:07+02:00"),
                unix_from_rfc3339("2026-10-23T11:43:07.5-04:30"),
                // Offsets that move the UTC time across a day, a leap day, and a year.
                unix_from_rfc3339("2026-10-24T01:13:07+09:00"),
                unix_from_rfc3339("2024-02-29T23:30:00-01:00"),
                unix_from_rfc3339("2100-02-28T23:00:00-02:00"),
                unix_from_rfc3339("2026-12-31T23:59:59-00:01"),
            ],
            [
                Some(0),
                Some(951_868_800),
                Some(1_792_771_987),
                Some(1_792_771_987),
                Some(1_792_771_987),
                Some(1_792_771_987),
                Some(1_709_253_000),
                Some(4_107_546_000),
                Some(1_798_761_659),
            ]
        );
    }

    #[test]
    fn malformed_rfc3339_timestamps_yield_none() {
        assert_eq!(
            [
                unix_from_rfc3339("garbage"),
                unix_from_rfc3339(""),
                unix_from_rfc3339("2026-10-23"),
                unix_from_rfc3339("2026-10-23T16:13:07"),
                unix_from_rfc3339("2026-13-23T16:13:07Z"),
                unix_from_rfc3339("2026-10-23T16:13:07.Z"),
                unix_from_rfc3339("2026-10-23T16:13:07+0200"),
                unix_from_rfc3339("2026-10-23T16:13:07\u{e9}"),
                unix_from_rfc3339("2026-10-23T16:13:07+24:00"),
                unix_from_rfc3339("2026-10-23T16:13:07+02:60"),
            ],
            [None; 10]
        );
    }

    #[tokio::test]
    async fn a_profile_without_credential_process_yields_no_credentials() {
        assert_eq!(
            [
                credentials_from_process("b", EXAMPLE_CONFIG).await,
                credentials_from_process("missing", EXAMPLE_CONFIG).await,
            ],
            [Ok(None), Ok(None)]
        );
    }

    #[tokio::test]
    async fn missing_aws_files_yield_the_plain_error() {
        let directory = tempfile::tempdir().expect("temp dir");

        assert_eq!(
            profile_credentials(directory.path(), "p").await,
            Err("No AWS credentials available for Bedrock".to_string())
        );
    }

    #[cfg(unix)]
    mod process {
        use std::time::{SystemTime, UNIX_EPOCH};

        use super::super::{
            civil_from_unix, credentials_from_process, profile_credentials, AwsCredentials,
        };

        /// The start of every error that names the profile `p`.
        const PROFILE_P_FAILURE: &str =
            "No AWS credentials available for Bedrock: credential_process for profile 'p' ";

        fn example_credentials() -> AwsCredentials {
            AwsCredentials {
                access_key_id: "AKIDEXAMPLE".to_string(),
                secret_access_key: "process-secret-example".to_string(),
                session_token: Some("session-token-example".to_string()),
            }
        }

        /// A shell command that prints example credentials. `fields` is spliced in before the
        /// closing brace, for example `,"Expiration":"2099-01-01T00:00:00Z"`.
        fn print_credentials(fields: &str) -> String {
            format!(
                r#"printf '{{"Version":1,"AccessKeyId":"AKIDEXAMPLE","SecretAccessKey":"process-secret-example","SessionToken":"session-token-example"{fields}}}'"#
            )
        }

        /// Config text whose profile `p` runs `command` as its `credential_process`.
        fn config_running(command: &str) -> String {
            format!("[profile p]\ncredential_process = {command}\n")
        }

        #[tokio::test]
        async fn credentials_are_read_from_the_credential_process() {
            let command = print_credentials(r#","Expiration":"2099-01-01T00:00:00Z""#);

            assert_eq!(
                credentials_from_process("p", &config_running(&command)).await,
                Ok(Some(example_credentials()))
            );
        }

        #[tokio::test]
        async fn a_profile_with_only_a_credential_process_resolves_through_the_config_file() {
            let directory = tempfile::tempdir().expect("temp dir");
            std::fs::write(
                directory.path().join("config"),
                config_running(&print_credentials("")),
            )
            .expect("config file");

            assert_eq!(
                profile_credentials(directory.path(), "p").await,
                Ok(example_credentials())
            );
        }

        #[tokio::test]
        async fn static_keys_in_the_credentials_file_come_before_the_credential_process() {
            let directory = tempfile::tempdir().expect("temp dir");
            std::fs::write(
                directory.path().join("credentials"),
                "[p]\naws_access_key_id = AKIDSTATIC\naws_secret_access_key = static-example\n",
            )
            .expect("credentials file");
            std::fs::write(directory.path().join("config"), config_running("exit 1"))
                .expect("config file");

            assert_eq!(
                profile_credentials(directory.path(), "p").await,
                Ok(AwsCredentials {
                    access_key_id: "AKIDSTATIC".to_string(),
                    secret_access_key: "static-example".to_string(),
                    session_token: None,
                })
            );
        }

        #[tokio::test]
        async fn an_unreadable_aws_file_is_the_error_only_when_no_source_has_credentials() {
            // A directory in place of a file fails to read with an error other than NotFound.
            let unreadable = std::fs::read_to_string(std::env::temp_dir())
                .expect_err("a directory does not read as a file");
            let credentials_is_a_directory = tempfile::tempdir().expect("temp dir");
            std::fs::create_dir(credentials_is_a_directory.path().join("credentials"))
                .expect("credentials directory");
            let config_is_a_directory = tempfile::tempdir().expect("temp dir");
            std::fs::create_dir(config_is_a_directory.path().join("config"))
                .expect("config directory");
            let process_fallback = tempfile::tempdir().expect("temp dir");
            std::fs::create_dir(process_fallback.path().join("credentials"))
                .expect("credentials directory");
            std::fs::write(
                process_fallback.path().join("config"),
                config_running(&print_credentials("")),
            )
            .expect("config file");

            assert_eq!(
                [
                    profile_credentials(credentials_is_a_directory.path(), "p").await,
                    profile_credentials(config_is_a_directory.path(), "p").await,
                    profile_credentials(process_fallback.path(), "p").await,
                ],
                [
                    Err(format!(
                        "No AWS credentials available for Bedrock: could not read {}: {unreadable}",
                        credentials_is_a_directory
                            .path()
                            .join("credentials")
                            .display()
                    )),
                    Err(format!(
                        "No AWS credentials available for Bedrock: could not read {}: {unreadable}",
                        config_is_a_directory.path().join("config").display()
                    )),
                    Ok(example_credentials()),
                ]
            );
        }

        #[tokio::test]
        async fn a_failing_credential_process_names_the_profile_and_the_first_stderr_line() {
            let results = [
                credentials_from_process("p", &config_running("echo boom >&2; exit 1")).await,
                credentials_from_process(
                    "p",
                    &config_running(r"printf '\n  \n boom \nlater\n' >&2; exit 1"),
                )
                .await,
                credentials_from_process("p", &config_running("exit 3")).await,
            ];

            assert_eq!(
                results,
                [
                    Err(format!(
                        "{PROFILE_P_FAILURE}exited with exit status: 1: boom"
                    )),
                    Err(format!(
                        "{PROFILE_P_FAILURE}exited with exit status: 1: boom"
                    )),
                    Err(format!("{PROFILE_P_FAILURE}exited with exit status: 3")),
                ]
            );
        }

        #[tokio::test]
        async fn a_long_stderr_line_is_cut_to_200_characters() {
            let result = credentials_from_process(
                "p",
                &config_running("head -c 300 /dev/zero | tr '\\0' x >&2; exit 1"),
            )
            .await;

            assert_eq!(
                result,
                Err(format!(
                    "{PROFILE_P_FAILURE}exited with exit status: 1: {}",
                    "x".repeat(200)
                ))
            );
        }

        #[tokio::test]
        async fn expired_credentials_are_rejected() {
            let command = print_credentials(r#","Expiration":"2000-01-01T00:00:00Z""#);

            assert_eq!(
                credentials_from_process("p", &config_running(&command)).await,
                Err(format!("{PROFILE_P_FAILURE}returned expired credentials"))
            );
        }

        #[tokio::test]
        async fn credential_process_output_that_is_not_version_1_credentials_is_rejected() {
            let cases = [
                (
                    r#"printf '{"Version":2,"AccessKeyId":"AKIDEXAMPLE","SecretAccessKey":"secret"}'"#,
                    "did not return Version 1",
                ),
                (
                    r#"printf '{"AccessKeyId":"AKIDEXAMPLE","SecretAccessKey":"secret"}'"#,
                    "did not return Version 1",
                ),
                (
                    r#"printf '{"Version":1,"SecretAccessKey":"secret"}'"#,
                    "returned invalid credentials",
                ),
                (
                    r#"printf '{"Version":1,"AccessKeyId":"AKIDEXAMPLE","SecretAccessKey":""}'"#,
                    "returned invalid credentials",
                ),
                (
                    r#"printf '{"Version":1,"AccessKeyId":"AKIDEXAMPLE","SecretAccessKey":"secret","SessionToken":7}'"#,
                    "returned invalid JSON",
                ),
                ("printf 'not json'", "returned invalid JSON"),
                ("true", "returned invalid JSON"),
            ];
            let mut results = Vec::new();
            for (command, _) in cases {
                results.push(credentials_from_process("p", &config_running(command)).await);
            }

            assert_eq!(
                results,
                cases
                    .map(|(_, detail)| Err(format!("{PROFILE_P_FAILURE}{detail}")))
                    .to_vec()
            );
        }

        /// Resolve twice with a `credential_process` that prints the example credentials plus
        /// `fields`, and count how often the process ran.
        async fn resolve_twice(
            fields: &str,
        ) -> ([Result<Option<AwsCredentials>, String>; 2], usize) {
            let directory = tempfile::tempdir().expect("temp dir");
            let log = directory.path().join("runs.log");
            let command = format!(
                "echo run >> '{}'; {}",
                log.display(),
                print_credentials(fields)
            );
            let config = config_running(&command);
            let results = [
                credentials_from_process("p", &config).await,
                credentials_from_process("p", &config).await,
            ];
            let runs = std::fs::read_to_string(&log)
                .expect("run log")
                .lines()
                .count();
            (results, runs)
        }

        #[tokio::test]
        async fn credentials_with_a_distant_expiration_are_cached_per_command() {
            assert_eq!(
                resolve_twice(r#","Expiration":"2099-01-01T00:00:00Z""#).await,
                (
                    [
                        Ok(Some(example_credentials())),
                        Ok(Some(example_credentials()))
                    ],
                    1
                )
            );
        }

        #[tokio::test]
        async fn credentials_without_an_expiration_are_never_cached() {
            assert_eq!(
                resolve_twice("").await,
                (
                    [
                        Ok(Some(example_credentials())),
                        Ok(Some(example_credentials()))
                    ],
                    2
                )
            );
        }

        #[tokio::test]
        async fn an_unparseable_expiration_counts_as_no_expiration() {
            assert_eq!(
                resolve_twice(r#","Expiration":"tomorrow""#).await,
                (
                    [
                        Ok(Some(example_credentials())),
                        Ok(Some(example_credentials()))
                    ],
                    2
                )
            );
        }

        #[tokio::test]
        async fn credentials_that_expire_within_five_minutes_are_not_served_from_the_cache() {
            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("clock after the epoch")
                .as_secs();
            let (year, month, day, hour, minute, second) = civil_from_unix(now + 100);
            let expiration =
                format!("{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}Z");

            assert_eq!(
                resolve_twice(&format!(r#","Expiration":"{expiration}""#)).await,
                (
                    [
                        Ok(Some(example_credentials())),
                        Ok(Some(example_credentials()))
                    ],
                    2
                )
            );
        }
    }
}
