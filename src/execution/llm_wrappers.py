import os
import json
import base64
from abc import ABC, abstractmethod
import time
import asyncio

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("Attempted to load .env file for wrappers.")
except ImportError:
    print("dotenv package not found, skipping .env file loading.")

# Import SDKs - check for ASYNC versions
try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None
try:
    from anthropic import AsyncAnthropic
except ImportError:
    AsyncAnthropic = None
try:
    # Use v1beta for Gemini 1.5 Flash/Pro
    import google.generativeai as genai
except ImportError:
    genai = None
try:
    from groq import AsyncGroq
except ImportError:
    AsyncGroq = None

# --- Helper: Image Loading ---
def encode_image_to_base64(image_path):
    """Reads an image file and encodes it to base64."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None
    except Exception as e:
        print(f"Error encoding image {image_path}: {e}")
        return None

# --- Base Class ---
class LLMWrapper(ABC):
    """Abstract base class for LLM API wrappers (Async)."""
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.api_key = self._get_api_key_from_env()
        if not self.api_key:
            raise ValueError(f"API key env var {self._get_env_var_name()} not found.")
        self._initialize_client()
        print(f"{self.__class__.__name__} initialized for model: {self.model_name}")

    @abstractmethod
    def _get_api_key_from_env(self) -> str | None:
        pass

    @abstractmethod
    def _get_env_var_name(self) -> str:
        pass

    @abstractmethod
    def _initialize_client(self):
        pass

    @abstractmethod
    async def generate(self, prompt: str, image_path: str | None = None) -> tuple[str | None, str | None]:
        """Sends prompt (and optional image) to LLM, returns (output_str, error_message)."""
        pass

    def _parse_response(self, raw_response: str) -> tuple[str | None, str | None]:
        """Attempts to parse the LLM response as JSON. Allows non-JSON pass-through."""
        if not raw_response:
            return None, "Received empty response from LLM."

        # Try to clean markdown code blocks if present
        cleaned_response = raw_response.strip()
        if cleaned_response.startswith("```json"):
            cleaned_response = cleaned_response[7:].strip()
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3].strip()
        elif cleaned_response.startswith("```"):
            cleaned_response = cleaned_response[3:].strip()
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3].strip()
            
        try:
            # Attempt JSON parsing
            parsed = json.loads(cleaned_response)
            # Return the parsed dict as a JSON string for consistency downstream
            return json.dumps(parsed), None
        except json.JSONDecodeError:
            # If it's not valid JSON, maybe it's just text. Return as is.
            # Add a warning, as we usually expect JSON.
            print(f"Warning: LLM response was not valid JSON. Passing through raw text: {raw_response[:150]}...")
            return raw_response, None # Return raw text, no error
        except Exception as e:
            return None, f"Unexpected error parsing response: {e}\nRaw response: {raw_response[:500]}..."

# --- Concrete Implementations ---

class OpenAIWrapper(LLMWrapper):
    def _get_env_var_name(self) -> str: return "OPENAI_API_KEY"
    def _get_api_key_from_env(self) -> str | None: return os.environ.get(self._get_env_var_name())

    def _initialize_client(self):
        if not AsyncOpenAI:
            raise ImportError("AsyncOpenAI not found. Install with: pip install --upgrade openai")
        self.async_client = AsyncOpenAI(api_key=self.api_key)

    async def generate(self, prompt: str, image_path: str | None = None) -> tuple[str | None, str | None]:
        task_id = asyncio.current_task().get_name() if asyncio.current_task() else 'main'
        print(f"Task {task_id}: OpenAI Request (Model: {self.model_name}) Image: {image_path is not None}")
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant designed to output JSON unless otherwise specified."}
        ]
        
        user_content = [{"type": "text", "text": prompt}]
        
        if image_path:
            image_base64 = encode_image_to_base64(image_path)
            if image_base64:
                user_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_base64}", # Assume PNG
                        "detail": "high" 
                    }
                })
            else:
                return None, f"Failed to encode image: {image_path}"
                
        messages.append({"role": "user", "content": user_content})

        try:
            if "o1" in self.model_name or "o3" in self.model_name or "o4" in self.model_name:
                api_args = {
                    "model": self.model_name,
                    "messages": messages,
                    "max_completion_tokens": 64000, # Increased max tokens for reasoning and potentially detailed JSON
                    "reasoning_effort": "high",
                }
            else:
                api_args = {
                    "model": self.model_name,
                    "messages": messages,
                    "max_tokens": 4096, # Increased max tokens for potentially detailed JSON
                    "temperature": 0, # Keep deterministic for benchmarks
                }
            # GPT-4o models might expect JSON object format slightly differently or have other params
            api_args["response_format"] = {"type": "json_object"}
            print(f"Task {task_id}: Using json_object response format.")

            response = await self.async_client.chat.completions.create(**api_args)
            raw_output = response.choices[0].message.content
            print(f"Task {task_id}: Received response from OpenAI.")
            return self._parse_response(raw_output)
        except Exception as e:
            error_message = f"Error calling OpenAI: {type(e).__name__}: {e}"
            print(f"Task {task_id}: {error_message}")
            return None, error_message

class AnthropicWrapper(LLMWrapper):
    def _get_env_var_name(self) -> str: return "ANTHROPIC_API_KEY"
    def _get_api_key_from_env(self) -> str | None: return os.environ.get(self._get_env_var_name())

    def _initialize_client(self):
        if not AsyncAnthropic:
             raise ImportError("AsyncAnthropic not found. Install with: pip install --upgrade anthropic")
        self.async_client = AsyncAnthropic(api_key=self.api_key)

    async def generate(self, prompt: str, image_path: str | None = None) -> tuple[str | None, str | None]:
        task_id = asyncio.current_task().get_name() if asyncio.current_task() else 'main'
        print(f"Task {task_id}: Anthropic Request (Model: {self.model_name}) Image: {image_path is not None}")

        system_prompt = "You are a helpful assistant designed to output JSON unless otherwise specified. Respond ONLY with the JSON structure requested in the prompt."
        user_content = [
            {"type": "text", "text": prompt}
        ]

        if image_path:
            image_base64 = encode_image_to_base64(image_path)
            if image_base64:
                user_content.insert(0, { # Anthropic often prefers image first
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png", # Assume PNG
                        "data": image_base64,
                    }
                })
            else:
                return None, f"Failed to encode image: {image_path}"

        messages = [
            {"role": "user", "content": user_content}
        ]

        try:
            # Add prefill for JSON if applicable (helps guide the model)
            # messages.append({"role": "assistant", "content": "{"}) 
            
            response = await self.async_client.messages.create(
                model=self.model_name,
                system=system_prompt,
                messages=messages,
                max_tokens=4096, # Increased max tokens
                temperature=0
            )
            raw_output = response.content[0].text
            print(f"Task {task_id}: Received response from Anthropic.")
            return self._parse_response(raw_output)
        except Exception as e:
            error_message = f"Error calling Anthropic: {type(e).__name__}: {e}"
            print(f"Task {task_id}: {error_message}")
            return None, error_message

class GoogleWrapper(LLMWrapper):
    def _get_env_var_name(self) -> str: return "GOOGLE_API_KEY"
    def _get_api_key_from_env(self) -> str | None: return os.environ.get(self._get_env_var_name())

    def _initialize_client(self):
        if not genai:
             raise ImportError("Google Generative AI SDK not found. Install with: pip install google-generativeai")
        try:
            genai.configure(api_key=self.api_key)
            # Initialize later in generate, as model needs to know if vision is required
            self.client = None 
        except Exception as e:
            raise ValueError(f"Failed to configure Google API key: {e}")

    async def generate(self, prompt: str, image_path: str | None = None) -> tuple[str | None, str | None]:
        task_id = asyncio.current_task().get_name() if asyncio.current_task() else 'main'
        print(f"Task {task_id}: Google Request (Model: {self.model_name}) Image: {image_path is not None}")

        # Use the model name provided during initialization directly
        model_name_to_use = self.model_name
        contents = [prompt]

        if image_path:
            image_base64 = encode_image_to_base64(image_path)
            if image_base64:
                image_part = {
                    "mime_type": "image/png", # Assume PNG
                    "data": image_base64
                }
                contents.insert(0, image_part) # Add image before prompt
            else:
                return None, f"Failed to encode image: {image_path}"
        
        # Initialize model using the provided name
        try:
            self.client = genai.GenerativeModel(model_name_to_use)
        except Exception as e:
             return None, f"Failed to initialize Google model {model_name_to_use}: {e}"
             
        generation_config = genai.types.GenerationConfig(
            temperature=0.0, 
            # Potentially force JSON output if model supports it
            response_mime_type="application/json" 
        )
        
        try:
            response = await self.client.generate_content_async(
                contents=contents,
                generation_config=generation_config,
                # safety_settings=safety_settings # Optional: relax safety settings if needed
            )
            # Handle potential blocks or empty responses
            if not response.candidates:
                 block_reason = response.prompt_feedback.block_reason if response.prompt_feedback else "Unknown"
                 error_message = f"Google response blocked or empty. Reason: {block_reason}"
                 print(f"Task {task_id}: {error_message}")
                 # print(f"Full Response Feedback: {response.prompt_feedback}") # More detail
                 return None, error_message
                 
            raw_output = response.text
            print(f"Task {task_id}: Received response from Google.")
            # Google might directly return JSON if response_mime_type worked
            # Our parser handles both JSON string and raw text that needs parsing
            return self._parse_response(raw_output) 
        except Exception as e:
            # Catch specific API errors if possible (e.g., deadline exceeded, resource exhausted)
            error_message = f"Error calling Google: {type(e).__name__}: {e}"
            # Check if the error message indicates a content policy issue
            if "response was blocked" in str(e).lower():
                error_message += " (Potentially due to safety filters)"
            print(f"Task {task_id}: {error_message}")
            # print(f"Full Response: {getattr(e, 'response', None)}") # For debugging API errors
            return None, error_message

class GroqWrapper(LLMWrapper):
    def _get_env_var_name(self) -> str: return "GROQ_API_KEY"
    def _get_api_key_from_env(self) -> str | None: return os.environ.get(self._get_env_var_name())

    def _initialize_client(self):
        if not AsyncGroq:
            raise ImportError("AsyncGroq not found. Install with: pip install --upgrade groq")
        self.async_client = AsyncGroq(api_key=self.api_key)

    async def generate(self, prompt: str, image_path: str | None = None) -> tuple[str | None, str | None]:
        task_id = asyncio.current_task().get_name() if asyncio.current_task() else 'main'
        print(f"Task {task_id}: Groq Request (Model: {self.model_name}) Image: {image_path is not None}")

        if image_path:
            # Groq models (currently) do not support image input via their API
            error_message = f"Groq models do not support image input. Skipping test case with image: {image_path}"
            print(f"Task {task_id}: {error_message}")
            return None, error_message
            
        messages = [
            {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
            {"role": "user", "content": prompt}
        ]

        try:
            response = await self.async_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0,
                max_tokens=4096,
                response_format={"type": "json_object"} # Request JSON
            )
            raw_output = response.choices[0].message.content
            print(f"Task {task_id}: Received response from Groq.")
            return self._parse_response(raw_output)
        except Exception as e:
            error_message = f"Error calling Groq: {type(e).__name__}: {e}"
            print(f"Task {task_id}: {error_message}")
            return None, error_message

# --- Factory Function ---
def get_llm_wrapper(provider: str, model_name: str) -> LLMWrapper:
    """Factory function to get the appropriate async LLM wrapper."""
    provider_lower = provider.lower()
    if provider_lower == "openai":
        return OpenAIWrapper(model_name=model_name)
    elif provider_lower == "anthropic":
        return AnthropicWrapper(model_name=model_name)
    elif provider_lower == "google":
        return GoogleWrapper(model_name=model_name)
    elif provider_lower == "groq":
        return GroqWrapper(model_name=model_name)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}") 