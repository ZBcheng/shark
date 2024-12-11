use std::{collections::HashMap, sync::Arc};

use minijinja::{context, Environment};
use ollama_rs::{
    generation::{
        chat::{ChatMessage, ChatMessageResponse},
        completion::{request::GenerationRequest, GenerationResponseStream},
        functions::{tools::Tool, DDGSearcher, FunctionCallRequest, LlamaFunctionCall},
    },
    Ollama,
};
use serde::Deserialize;

use crate::tools::rust_toolchain_switcher::RustToolchainSwitcher;

const SHARK_FUNCTION_CALLING_PROMPT_TEMPLATE: &'static str = r#"
You are a helpful assistant, you can decide whether to use functions to answer user's question: {{question}}
Here are the names and descriptions of the functions you can use:
{{functionss_description}}

If you decide to use a function, please response in the following format:
{
    "function": {name of the function}
}

If you don't want to use any functions, please response in the follow format:
{
    "function": null
}
"#;

const SHARK_GENERATATION_PROMPT_TEMPLATE: &'static str = r#"
You are a helpful assistant called shark🦈, answer the question given by user: {{question}}
"#;

const SHARK_SUMMARIZING_PROMPT_TEMPLATE: &'static str = r#"
You are a helpful assistant called shark🦈, given user's question: {{question}} and the answer of the question: {{answer}}, try to give a short summary.
Just response your summary content.
"#;

type Error = Box<dyn std::error::Error + 'static>;

pub struct Shark<'a> {
    core: Ollama,
    model: String,
    functions: HashMap<String, Arc<dyn Tool>>,
    template_env: Environment<'a>,
}

impl<'a> Shark<'a> {
    pub fn new(core: Ollama, model: impl ToString, functions: Vec<String>) -> Self {
        let mut template_env = Environment::new();
        template_env
            .add_template("function-calling", SHARK_FUNCTION_CALLING_PROMPT_TEMPLATE)
            .unwrap();

        template_env
            .add_template("generation", SHARK_GENERATATION_PROMPT_TEMPLATE)
            .unwrap();

        template_env
            .add_template("summary", SHARK_SUMMARIZING_PROMPT_TEMPLATE)
            .unwrap();

        Self {
            core,
            model: model.to_string(),
            functions: Self::parse_functions(functions),
            template_env,
        }
    }

    pub async fn generate_stream(
        &self,
        question: impl ToString,
    ) -> Result<GenerationResponseStream, Error> {
        let question = question.to_string();
        let function = self.request_function(&question).await?;
        if let Some(func) = function {
            let function_calling_response = self.call_function(&question, func).await?;
            let response = function_calling_response.message.unwrap().content;
            let stream = self.summarize_stream(question, response).await?;
            Ok(stream)
        } else {
            let template = self.template_env.get_template("generation").unwrap();
            let prompt = template.render(context! {question => question})?;
            let stream = self
                .core
                .generate_stream(GenerationRequest::new(self.model.to_owned(), prompt))
                .await?;
            return Ok(stream);
        }
    }

    async fn call_function(
        &self,
        question: impl ToString,
        func: Arc<dyn Tool>,
    ) -> Result<ChatMessageResponse, Error> {
        let parser = Arc::new(LlamaFunctionCall {});
        let message = ChatMessage::user(question.to_string());
        let response = self
            .core
            .send_function_call(
                FunctionCallRequest::new(self.model.to_owned(), vec![func], vec![message]),
                parser,
            )
            .await?;

        Ok(response)
    }

    async fn summarize_stream(
        &self,
        question: impl ToString,
        answer: impl ToString,
    ) -> Result<GenerationResponseStream, Error> {
        let (question, answer) = (question.to_string(), answer.to_string());
        let template = self.template_env.get_template("summary").unwrap();
        let prompt = template.render(context! {question => question, answer => answer})?;
        let stream = self
            .core
            .generate_stream(GenerationRequest::new(self.model.to_owned(), prompt))
            .await?;
        Ok(stream)
    }

    async fn request_function(
        &self,
        question: impl ToString,
    ) -> Result<Option<Arc<dyn Tool>>, Error> {
        let question = question.to_string();
        let functionss_description = self.functions_description();
        let template = self.template_env.get_template("function-calling")?;
        let prompt = template.render(
            context! {question => question, functionss_description => functionss_description},
        )?;

        let function_resp = self
            .core
            .generate(GenerationRequest::new(self.model.to_owned(), prompt))
            .await?
            .response;

        let response: FucntionResponse = serde_json::from_str(&function_resp)?;
        if response.function.is_none() {
            return Ok(None);
        }

        let function_name = response
            .function
            .unwrap()
            .trim()
            .to_lowercase()
            .replace("'", "");

        if function_name == "null" {
            return Ok(None);
        }

        if let Some(f) = self.functions.get(&function_name) {
            return Ok(Some(f.to_owned()));
        }

        Ok(None)
    }

    fn functions_description(&self) -> String {
        let desc: HashMap<String, String> = self
            .functions
            .iter()
            .map(|(name, tool)| (name.to_owned(), tool.description()))
            .collect();

        serde_json::to_string(&desc).unwrap()
    }

    fn parse_functions(functions: Vec<String>) -> HashMap<String, Arc<dyn Tool>> {
        let mut function_set: HashMap<String, Arc<dyn Tool>> = HashMap::new();
        for f in functions {
            let function_name = f.trim().to_lowercase();
            match function_name.as_str() {
                "ddg_searcher" => {
                    function_set.insert("ddg_searcher".to_string(), Arc::new(DDGSearcher::new()));
                }
                "rust_toolchain_switcher" => {
                    function_set.insert(
                        "rust_toolchain_switcher".to_string(),
                        Arc::new(RustToolchainSwitcher::default()),
                    );
                }
                other => println!("unknown tool: {other}"),
            }
        }

        function_set
    }
}

#[derive(Debug, Deserialize)]
struct FucntionResponse {
    function: Option<String>,
}
