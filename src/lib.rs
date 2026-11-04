//! squeeze: cut Claude Code token spend with a local, cache-aware compression
//! proxy and token-efficient MCP tools.
//!
//! Not affiliated with or endorsed by Anthropic.

pub mod atlas;
pub mod auth;
pub mod bench;
pub mod compress;
pub mod hook;
pub mod mcp;
pub mod outline;
pub mod proxy;
pub mod replay;
pub mod sse;
pub mod store;
pub mod tools;
pub mod transform;
pub mod util;

pub const VERSION: &str = env!("CARGO_PKG_VERSION");
