import httpx
import jwt
import json
import logging
import os
import time
from src.command_executor import execute_command_in_terminal, execute_command
from src.utils import load_persistent_memory
from src.mcp_protocol import mcp  # Import the MCP singleton
import openai
from src.token_manager import TokenManager
from src.command_executor import wait_for_command_completion
from src.approval_handler import ApprovalHandler

# Clear all proxy environment variables
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)
os.environ.pop("all_proxy", None)
os.environ.pop("ALL_PROXY", None)
os.environ.pop("socks_proxy", None)
os.environ.pop("SOCKS_PROXY", None)

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)


class NeoAI:
    def __init__(self, config):
        self.mode = config.get("mode", "lm_studio")
        logging.info(f"Initializing NeoAI in {self.mode} mode.")
        self.require_approval = config.get("command_approval", {}).get(
            "require_approval", True
        )
        self.auto_approve_all = config.get("command_approval", {}).get(
            "auto_approve_all", False
        )
        self.is_streaming_mode = config.get("stream", True)
        self.config = config

        if self.mode == "digital_ocean":
            do_cfg = config.get("digital_ocean_config", {})
            self.token_manager = TokenManager(
                agent_id=do_cfg.get("agent_id"),
                agent_key=do_cfg.get("agent_key"),
                auth_api_url="https://cluster-api.do-ai.run/v1",
            )
            self.access_token = self.token_manager.get_valid_access_token()
            self.token_timestamp = time.time()
            self.agent_endpoint = do_cfg.get("agent_endpoint")
            self.model = do_cfg.get("model")
        else:
            openai.api_base = config.get("api_url")
            openai.api_key = config.get("api_key")
            self.model = config.get("model")

        self.lm_studio_config = config.get("lm_studio_config", {})
        self.history = []

        # Inject pre-prompt instructions so the language model knows about the
        # Machine Communication Protocol (MCP).  Without this, local models may
        # respond like a regular chat bot and never issue command tags.
        preprompt_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "config",
            "PrePromt.md",
        )
        if os.path.exists(preprompt_path):
            try:
                with open(preprompt_path, "r") as f:
                    preprompt_text = f.read()
                self.history.append({"role": "system", "content": preprompt_text})
            except Exception as e:
                logging.error(f"Failed to load pre-prompt: {e}")

        self.context_initialized = False

    def _ensure_valid_token(self):
        """Check that the token is still valid and renew it if necessary"""
        if self.mode != "digital_ocean":
            return

        # Check token every 15 minutes or in case of 401 error
        current_time = time.time()
        token_age = current_time - self.token_timestamp

        if token_age > 900:  # 15 minutes
            try:
                logging.info("Token age > 15 minutes, refreshing...")
                self.access_token = self.token_manager.get_valid_access_token()
                self.token_timestamp = current_time
            except Exception as e:
                logging.error(f"Error refreshing token: {e}")
                # Attempt to completely reset token management
                do_cfg = self.config.get("digital_ocean_config", {})
                self.token_manager = TokenManager(
                    agent_id=do_cfg.get("agent_id"),
                    agent_key=do_cfg.get("agent_key"),
                    auth_api_url="https://cluster-api.do-ai.run/v1",
                )
                self.access_token = self.token_manager.get_valid_access_token()
                self.token_timestamp = current_time

    def initialize_context(self):
        context_commands = ["pwd", "ls"]
        context_data = load_persistent_memory()
        initial_context = "<context>\n"

        for command in context_commands:
            result = execute_command(command)
            initial_context += f"Command: {command}\nResult:\n{result}\n"

        full_context = f"{context_data}\n\n{initial_context}</context>"
        self.context_initialized = True
        return full_context

    def _query_lm_studio(self, prompt, clear_thinking=False):
        instruction = f"{self.lm_studio_config.get('input_prefix', '### Instruction:')} {prompt} {self.lm_studio_config.get('input_suffix', '### Response:')}"

        messages = []

        # LM Studio models typically only understand 'user' and 'assistant'
        # message roles. Map all other roles (e.g. 'system', 'tool') to 'user'
        # before sending the history to avoid template errors.
        for msg in self.history:
            role = msg.get("role", "user")
            if role not in ("user", "assistant"):
                role = "user"
            messages.append({"role": role, "content": msg.get("content", "")})

        messages.append({"role": "user", "content": instruction})

        try:
            completion = openai.ChatCompletion.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                stream=self.is_streaming_mode,
            )

            full_response = ""
            is_first_chunk = True

            for chunk in completion:
                if "choices" in chunk and len(chunk["choices"]) > 0:
                    content = chunk["choices"][0]["delta"].get("content", "")
                    if content:
                        if is_first_chunk:
                            if clear_thinking:
                                print("\r" + " " * 30 + "\r", end="", flush=True)
                            print("\033[1;34mNeo:\033[0m ", end="", flush=True)
                            is_first_chunk = False

                        print(content, end="", flush=True)
                        full_response += content

            print()
            return self._process_response(full_response)

        except Exception as e:
            print(f"Error while querying LM Studio: {e}")
            return "An error occurred while querying LM Studio."

    def _query_digitalocean(self, prompt, clear_thinking=False):
        self._ensure_valid_token()

        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": self.history + [{"role": "user", "content": prompt}],
            "stream": True,
        }

        max_retries = 2
        retry_count = 0

        while retry_count < max_retries:
            try:
                with httpx.stream(
                    "POST",
                    f"{self.agent_endpoint}/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=30.0,
                ) as response:
                    response.raise_for_status()

                    is_first_chunk = True
                    assistant_response = ""

                    for line in response.iter_lines():
                        line = line.strip()
                        if line.startswith("data:"):
                            line = line[len("data:") :].strip()

                        if line:
                            try:
                                chunk = json.loads(line)
                                if "choices" in chunk and chunk["choices"]:
                                    content = (
                                        chunk["choices"][0]
                                        .get("delta", {})
                                        .get("content", "")
                                    )
                                    if content:
                                        if is_first_chunk:
                                            if clear_thinking:
                                                print(
                                                    "\r" + " " * 30 + "\r",
                                                    end="",
                                                    flush=True,
                                                )
                                            print(
                                                "\033[1;34mNeo:\033[0m ",
                                                end="",
                                                flush=True,
                                            )
                                            is_first_chunk = False
                                        print(content, end="", flush=True)
                                        assistant_response += content
                            except json.JSONDecodeError:
                                if line == "[DONE]":
                                    break
                                continue
                    print()

                    if assistant_response.strip():
                        self.history.append(
                            {"role": "assistant", "content": assistant_response.strip()}
                        )
                        return self._process_response(assistant_response.strip())
                    return ""

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 401 and retry_count < max_retries:
                    retry_count += 1
                    # Force token refresh
                    do_cfg = self.config.get("digital_ocean_config", {})
                    self.token_manager = TokenManager(
                        agent_id=do_cfg.get("agent_id"),
                        agent_key=do_cfg.get("agent_key"),
                        auth_api_url="https://cluster-api.do-ai.run/v1",
                    )
                    self.access_token = self.token_manager.get_valid_access_token()
                    self.token_timestamp = time.time()

                    # Update header with new token
                    headers["Authorization"] = f"Bearer {self.access_token}"
                    continue
                else:
                    print(f"Details: {e}")
                    break

            except httpx.ReadTimeout:
                retry_count += 1
                if retry_count < max_retries:
                    self.access_token = self.token_manager.get_valid_access_token()
                    self.token_timestamp = time.time()
                    headers["Authorization"] = f"Bearer {self.access_token}"
                    continue
                else:
                    print("Failed after multiple attempts.")
                    break

            except Exception as e:
                import traceback

                print(f"\nDetailed error: {e}")
                print(traceback.format_exc())
                break

        return "Sorry, I couldn't get a response. Please try again."

    def query(self, prompt, clear_thinking=False):
        try:
            if not self.context_initialized:
                context = self.initialize_context()
                prompt = f"{context}\n\n{prompt}"

            self.history.append({"role": "user", "content": prompt})

            if self.mode == "digital_ocean":
                return self._query_digitalocean(prompt, clear_thinking)
            else:
                response = self._query_lm_studio(prompt, clear_thinking)
                if response:
                    self.history.append({"role": "assistant", "content": response})
                return response
        except Exception as e:
            import traceback

            print(f"Details: {e}")
            print(traceback.format_exc())

    def _process_response(self, response):
        """
        Process the AI response and handle MCP protocol commands.
        This replaces the old system tag processing with the new MCP protocol.

        Args:
            response: Text response from the AI

        Returns:
            Processed response with command outputs integrated
        """
        try:
            # Process all MCP protocol tags using the MCP singleton
            mcp_results = mcp.process_response(
                response,
                require_approval=self.require_approval,
                auto_approve=self.auto_approve_all,
            )

            # Log the MCP results for debugging
            # print(f"DEBUG - MCP results: {mcp_results}")

            # Check if any protocols were executed
            follow_up_messages = []

            for protocol, result in mcp_results.items():
                # Skip error key or non-dict results
                if protocol == "error" or not isinstance(result, dict):
                    continue

                # Check if the protocol was executed
                if result.get("executed", False):
                    command = result.get("command", "unknown command")
                    output = result.get("output", "No output")

                    # Create a follow-up message for this protocol
                    follow_up_prompt = f"The {protocol} command '{command}' was executed. Here is the result:\n{output}"
                    follow_up_messages.append(follow_up_prompt)

            # If we have follow-up messages, send them to the AI
            if follow_up_messages:
                combined_prompt = "\n\n".join(follow_up_messages)
                self.history.append({"role": "user", "content": combined_prompt})

                if self.mode == "digital_ocean":
                    return self._query_digitalocean(combined_prompt)
                elif self.mode == "lm_studio":
                    return self._query_lm_studio(combined_prompt)
                else:
                    return (
                        f"Unknown mode: {self.mode}. Unable to send follow-up prompt."
                    )

        except Exception as e:
            import traceback

            print(f"\nAn unexpected error occurred while processing the response: {e}")
            print(traceback.format_exc())
            return "An error occurred while processing the command."

        return response.strip()

    def get_conversation_history(self):
        return self.history

    def reset_history(self):
        self.history = []
        self.context_initialized = False
        self.auto_approve_all = False
