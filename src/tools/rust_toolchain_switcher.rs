use std::{collections::HashMap, error::Error};

use async_trait::async_trait;
use ollama_rs::generation::functions::tools::Tool;
use serde_json::{json, Value};
use tokio::process::Command;

#[derive(Default)]
pub struct RustToolchainSwitcher {}

#[async_trait]
impl Tool for RustToolchainSwitcher {
    fn name(&self) -> String {
        "rust_toolchain_switcher".to_string()
    }

    fn description(&self) -> String {
        "Switch rust toolchain between 'stable', 'nightly' and others".to_string()
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "toolchain": {
                    "type": "string",
                    "description": "The toolchain you want to switch"
                }
            },
            "required": ["toolchain"]
        })
    }

    async fn run(&self, input: Value) -> Result<String, Box<dyn Error>> {
        let version = input["toolchain"].as_str().unwrap();
        let output = Command::new("rustup")
            .args(["default", version])
            .output()
            .await?;

        let output_content = if output.stdout.len() > 0 {
            String::from_utf8(output.stdout)?
        } else {
            String::default()
        };

        let output_error = if output.stderr.len() > 0 {
            String::from_utf8(output.stderr)?
        } else {
            String::default()
        };

        let mut response = HashMap::new();
        response.insert("result", output_content);
        response.insert("error", output_error);

        Ok(serde_json::to_string(&response)?)
    }
}
