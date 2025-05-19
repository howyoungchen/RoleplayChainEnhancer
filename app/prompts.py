from typing import Dict, List, Any
from .utils import load_config, logger

class PromptManager:
    def __init__(self):
        try:
            config = load_config()
            self.prompts_config = config.get("prompts", {})
            if not self.prompts_config:
                logger.warning("No prompts found in config.yml")
            
            self.system_prompt: str = self.prompts_config.get("system", "").strip()
            self.user_part1: str = self.prompts_config.get("user_part1", "").strip()
            self.user_part2: str = self.prompts_config.get("user_part2", "").strip()

            if not self.system_prompt:
                logger.warning("System prompt is empty or not defined in config.yml")
            if not self.user_part1:
                 logger.warning("User part 1 prompt is empty or not defined in config.yml")
            if not self.user_part2:
                 logger.warning("User part 2 prompt is empty or not defined in config.yml")

        except FileNotFoundError:
            logger.error("prompts.py: Configuration file not found. Prompts will be empty.")
            self.prompts_config = {}
            self.system_prompt = ""
            self.user_part1 = ""
            self.user_part2 = ""
        except Exception as e:
            logger.error(f"prompts.py: Error loading prompts from config: {e}")
            self.prompts_config = {}
            self.system_prompt = ""
            self.user_part1 = ""
            self.user_part2 = ""

    def get_system_prompt(self) -> str:
        return self.system_prompt

    def get_user_part1(self) -> str:
        return self.user_part1

    def get_user_part2(self) -> str:
        return self.user_part2

    def construct_primary_llm_messages(self, original_user_content: str) -> List[Dict[str, Any]]:
        """
        Constructs the messages list for the primary LLM call (Step 4).
        The original_user_content is the concatenated content from the user's messages.
        """
        # Format specified:
        # [\n  {\"role\":\"system\",\"content\":\"<系统 prompt>\"},\n  {\"role\":\"user\",\"content\":\"<第一部分 user prompt>\\n======\\n<原始用户输入>\\n======\\n<第二部分 user prompt>\"}\n]\n
        
        # Ensure all parts are strings. If any part is missing, it should be an empty string.
        sys_prompt = self.get_system_prompt()
        usr_p1 = self.get_user_part1()
        usr_p2 = self.get_user_part2()

        # Construct the user content string
        # Using f-string for clarity, ensuring newlines are correctly placed.
        # The spec shows "\n======\n<原始用户输入>\n======\n"
        # Let's adjust to make sure it's robust if parts are empty.
        
        # If original_user_content is multi-line, it should be preserved.
        # The spec has <原始用户输入> directly.
        
        # The spec has: "content": "<第一部分 user prompt>\n======\n<原始用户输入>\n======\n<第二部分 user prompt>"
        # This implies that if user_part1 or user_part2 are empty, the "======\n" separators might still appear.
        # Let's refine this to be cleaner if parts are missing.

        user_content_parts = []
        if usr_p1:
            user_content_parts.append(usr_p1)
        
        # Separator only if both usr_p1 and original_user_content are present, or if original_user_content and usr_p2 are present.
        # Or, more simply, always include original_user_content, and separators if surrounding prompts exist.
        
        # Let's follow the structure: <part1>\n======\n<original>\n======\n<part2>
        # If a part is empty, it's just omitted. The separators should be conditional.

        full_user_content = ""
        if usr_p1:
            full_user_content += usr_p1
        
        if original_user_content:
            if full_user_content: # If usr_p1 was added
                full_user_content += "\n======\n"
            full_user_content += original_user_content
        
        if usr_p2:
            if full_user_content: # If usr_p1 or original_user_content was added
                full_user_content += "\n======\n"
            full_user_content += usr_p2
            
        messages = []
        if sys_prompt: # Only add system message if it's not empty
            messages.append({"role": "system", "content": sys_prompt})
        
        # User message should always be present, even if its content is built from empty parts
        # (though ideally, prompts are configured).
        messages.append({"role": "user", "content": full_user_content})
        
        logger.debug("Constructed primary LLM messages", messages=messages)
        return messages

# Instantiate the manager so it can be imported and used.
prompt_manager = PromptManager()

if __name__ == "__main__":
    # Test loading prompts
    print(f"System Prompt: '{prompt_manager.get_system_prompt()}'")
    print(f"User Part 1: '{prompt_manager.get_user_part1()}'")
    print(f"User Part 2: '{prompt_manager.get_user_part2()}'")

    test_original_content = "This is the original user input."
    constructed_messages = prompt_manager.construct_primary_llm_messages(test_original_content)
    print("\nConstructed Messages for Primary LLM:")
    import json
    print(json.dumps(constructed_messages, indent=2, ensure_ascii=False))

    test_original_content_empty = ""
    constructed_messages_empty = prompt_manager.construct_primary_llm_messages(test_original_content_empty)
    print("\nConstructed Messages for Primary LLM (empty original content):")
    print(json.dumps(constructed_messages_empty, indent=2, ensure_ascii=False))
    
    # Test with only system prompt
    # To do this, you'd need to modify config.yml temporarily or mock load_config
    # For now, this test assumes config.yml has all prompts.