use std::io::Write;

use clap::Parser;
use ollama_rs::{IntoUrlSealed, Ollama};
use serde::Deserialize;
use shark::Shark;
use termcolor::{Color, ColorChoice, ColorSpec, StandardStream, WriteColor};
use tokio_stream::StreamExt;
use toml;

pub mod shark;
pub mod tools;

#[derive(Debug, Parser)]
struct Args {
    prompt: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct Config {
    addr: String,
    model: String,
    color: String,
    functions: Vec<String>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config_path = std::env::var("CONFIG").expect("Missing CONFIG");
    let config = parse_config(&config_path);

    let args = Args::parse();
    let question = args.prompt.join(" ");

    let url = config.addr.into_url().unwrap();
    let ollama = Ollama::from_url(url);
    let shark = Shark::new(ollama, config.model, config.functions);

    let mut stdout = StandardStream::stdout(ColorChoice::Always);
    let mut color_spec = ColorSpec::new();

    stdout.set_color(color_spec.set_fg(Some(Color::Cyan)))?;
    let stream = shark.generate_stream(question).await;
    if let Err(e) = stream {
        stdout.set_color(color_spec.set_fg(Some(Color::Red)))?;
        let err =
            format!("Sorry I can't answer your question right now, please try again later.😭\n{e}");
        stdout.write(err.as_bytes())?;
        stdout.flush()?;
        return Ok(());
    }

    let mut stream = stream.unwrap();

    let color = parse_color(&config.color);
    stdout.set_color(color_spec.set_fg(Some(color)))?;
    while let Some(Ok(responses)) = stream.next().await {
        for resp in responses {
            stdout.write(resp.response.as_bytes())?;
            stdout.flush()?;
        }
    }

    stdout.write(b"\n")?;

    Ok(())
}

fn parse_config(path: &str) -> Config {
    let file = std::fs::read_to_string(path).unwrap();
    let config: Config = toml::from_str(&file).unwrap();
    config
}

fn parse_color(color: &str) -> Color {
    let color = color.trim().to_lowercase();
    match color.as_str() {
        "purple" => Color::Rgb(202, 158, 230),
        "red" => Color::Rgb(231, 130, 132),
        "green" => Color::Rgb(166, 209, 137),
        _ => Color::Green,
    }
}
