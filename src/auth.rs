//! Credential policy.
//!
//! Anthropic's terms do not allow third-party tools to route or intermediate
//! Claude subscription (Free/Pro/Max/Ultra) credentials:
//! <https://code.claude.com/docs/en/legal-and-compliance>. squeeze therefore
//! only proxies API-key or gateway-credential traffic, and refuses any request
//! that carries subscription OAuth credentials instead of forwarding it.

use axum::http::HeaderMap;
use axum::http::header::AUTHORIZATION;

pub const POLICY_URL: &str = "https://code.claude.com/docs/en/legal-and-compliance";

/// Prefix of Claude subscription OAuth access tokens.
const OAUTH_TOKEN_PREFIX: &str = "sk-ant-oat";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Refusal {
    SubscriptionOAuth,
}

/// Decide whether a request may be forwarded. Only header prefixes are
/// inspected; credential values are never logged or stored.
pub fn check_request(headers: &HeaderMap) -> Result<(), Refusal> {
    // Subscription requests carry an OAuth capability in anthropic-beta.
    let oauth_beta = headers
        .get_all("anthropic-beta")
        .iter()
        .filter_map(|v| v.to_str().ok())
        .flat_map(|v| v.split(','))
        .any(|b| b.trim().to_ascii_lowercase().starts_with("oauth-"));
    if oauth_beta {
        return Err(Refusal::SubscriptionOAuth);
    }
    if let Some(v) = headers.get(AUTHORIZATION).and_then(|v| v.to_str().ok())
        && is_oauth_token(strip_bearer(v))
    {
        return Err(Refusal::SubscriptionOAuth);
    }
    if let Some(v) = headers.get("x-api-key").and_then(|v| v.to_str().ok())
        && is_oauth_token(v.trim())
    {
        return Err(Refusal::SubscriptionOAuth);
    }
    Ok(())
}

pub fn is_oauth_token(token: &str) -> bool {
    token.starts_with(OAUTH_TOKEN_PREFIX)
}

fn strip_bearer(v: &str) -> &str {
    let v = v.trim();
    match v.get(..7) {
        Some(p) if p.eq_ignore_ascii_case("bearer ") => v[7..].trim(),
        _ => v,
    }
}

pub fn refusal_message(_: Refusal) -> String {
    format!(
        "squeeze refused to forward this request because it uses Claude subscription (OAuth) \
         credentials. Anthropic's terms do not allow third-party tools to route or intermediate \
         subscription credentials ({POLICY_URL}). squeeze works only with API-key or gateway \
         credentials: set ANTHROPIC_API_KEY (or ANTHROPIC_AUTH_TOKEN for a gateway), and approve \
         the key when Claude Code asks."
    )
}

/// The credential Claude Code will pick, following its documented precedence
/// (cloud provider, ANTHROPIC_AUTH_TOKEN, ANTHROPIC_API_KEY, apiKeyHelper,
/// CLAUDE_CODE_OAUTH_TOKEN, subscription login). apiKeyHelper lives in settings
/// files and cannot be seen here, so it shows up as `None`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Credential {
    CloudProvider(&'static str),
    AuthToken,
    ApiKey,
    /// An OAuth token supplied through the environment.
    OAuthEnv,
    None,
}

pub fn detect_credential(get: impl Fn(&str) -> Option<String>) -> Credential {
    let nonempty = |k: &str| get(k).filter(|v| !v.trim().is_empty());
    let truthy =
        |k: &str| nonempty(k).is_some_and(|v| !matches!(v.trim().to_ascii_lowercase().as_str(), "0" | "false" | "no"));
    for (var, name) in [
        ("CLAUDE_CODE_USE_BEDROCK", "Amazon Bedrock"),
        ("CLAUDE_CODE_USE_VERTEX", "Google Vertex AI"),
        ("CLAUDE_CODE_USE_FOUNDRY", "Microsoft Foundry"),
    ] {
        if truthy(var) {
            return Credential::CloudProvider(name);
        }
    }
    if let Some(t) = nonempty("ANTHROPIC_AUTH_TOKEN") {
        return if is_oauth_token(t.trim()) { Credential::OAuthEnv } else { Credential::AuthToken };
    }
    if let Some(k) = nonempty("ANTHROPIC_API_KEY") {
        return if is_oauth_token(k.trim()) { Credential::OAuthEnv } else { Credential::ApiKey };
    }
    if nonempty("CLAUDE_CODE_OAUTH_TOKEN").is_some() {
        return Credential::OAuthEnv;
    }
    Credential::None
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::HeaderValue;
    use std::collections::HashMap;

    fn headers(pairs: &[(&'static str, &str)]) -> HeaderMap {
        let mut h = HeaderMap::new();
        for (k, v) in pairs {
            h.append(*k, HeaderValue::from_str(v).unwrap());
        }
        h
    }

    #[test]
    fn refuses_oauth_beta() {
        let h = headers(&[("anthropic-beta", "interleaved-thinking-2025-05-14, oauth-2025-04-20")]);
        assert_eq!(check_request(&h), Err(Refusal::SubscriptionOAuth));
    }

    #[test]
    fn refuses_oauth_bearer_token() {
        let h = headers(&[("authorization", "Bearer sk-ant-oat01-abc")]);
        assert_eq!(check_request(&h), Err(Refusal::SubscriptionOAuth));
    }

    #[test]
    fn allows_api_key_and_gateway_token() {
        let h = headers(&[("x-api-key", "sk-ant-api03-abc"), ("anthropic-beta", "context-management-2025-06-27")]);
        assert_eq!(check_request(&h), Ok(()));
        let h = headers(&[("authorization", "Bearer gw-team-token")]);
        assert_eq!(check_request(&h), Ok(()));
    }

    fn env(pairs: &[(&str, &str)]) -> impl Fn(&str) -> Option<String> {
        let m: HashMap<String, String> = pairs.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect();
        move |k| m.get(k).cloned()
    }

    #[test]
    fn credential_precedence() {
        assert_eq!(detect_credential(env(&[])), Credential::None);
        assert_eq!(detect_credential(env(&[("ANTHROPIC_API_KEY", "sk-ant-api03-x")])), Credential::ApiKey);
        assert_eq!(
            detect_credential(env(&[("ANTHROPIC_API_KEY", "k"), ("ANTHROPIC_AUTH_TOKEN", "gw")])),
            Credential::AuthToken
        );
        assert_eq!(detect_credential(env(&[("CLAUDE_CODE_OAUTH_TOKEN", "sk-ant-oat01-x")])), Credential::OAuthEnv);
        assert_eq!(
            detect_credential(env(&[("ANTHROPIC_API_KEY", "k"), ("CLAUDE_CODE_OAUTH_TOKEN", "t")])),
            Credential::ApiKey
        );
        assert_eq!(
            detect_credential(env(&[("CLAUDE_CODE_USE_VERTEX", "1"), ("ANTHROPIC_API_KEY", "k")])),
            Credential::CloudProvider("Google Vertex AI")
        );
        assert_eq!(detect_credential(env(&[("CLAUDE_CODE_USE_BEDROCK", "0")])), Credential::None);
    }
}
