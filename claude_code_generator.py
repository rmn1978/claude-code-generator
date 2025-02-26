import os
import sys
import json
import time
import argparse
import anthropic
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

class ClaudeContinuousCodeGenerator:
    """A tool to generate large codebases using Claude 3.7 API incrementally."""
    
    def __init__(self, api_key: str, model: str = "claude-3-7-sonnet-20250219", 
                 max_tokens_per_request: int = 4000, temperature: float = 0.2,
                 output_dir: str = "generated_project"):
        """
        Initialize the code generator.
        
        Args:
            api_key: Anthropic API key
            model: Claude model to use
            max_tokens_per_request: Maximum tokens for each generation
            temperature: Sampling temperature (lower for more deterministic outputs)
            output_dir: Directory to save generated files
        """
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens_per_request
        self.temperature = temperature
        self.output_dir = Path(output_dir)
        self.conversation_history = []
        self.file_registry = {}  # Tracks created files and their content summaries
        
    def create_directory_structure(self) -> None:
        """Create the output directory if it doesn't exist."""
        if not self.output_dir.exists():
            os.makedirs(self.output_dir, exist_ok=True)
            print(f"Created output directory: {self.output_dir}")
        else:
            print(f"Using existing output directory: {self.output_dir}")

    def parse_code_blocks(self, text: str) -> List[Dict[str, str]]:
        """
        Parse code blocks in Claude's response.
        
        Args:
            text: Response text from Claude
            
        Returns:
            List of dictionaries containing file_path and code
        """
        code_blocks = []
        lines = text.split('\n')
        
        in_code_block = False
        current_block = {"file_path": None, "code": ""}
        file_path_line = ""
        
        for line in lines:
            # Check for file path indicators
            if not in_code_block and "```" not in line and ("File:" in line or line.endswith(".py") or 
                                                          line.endswith(".js") or line.endswith(".html") or
                                                          line.endswith(".css") or line.endswith(".json") or
                                                          "/" in line and "." in line.split("/")[-1]):
                file_path_line = line.replace("File:", "").strip()
                
            # Start of code block
            elif "```" in line and not in_code_block:
                in_code_block = True
                # If the code block has a language specifier, remove it
                language_marker = line.replace("```", "").strip()
                
                # If we found a file path before this code block
                if file_path_line:
                    current_block["file_path"] = file_path_line
                # If no explicit file path but language is specified, create a default filename
                elif language_marker:
                    extension = language_marker
                    if language_marker == "python":
                        extension = "py"
                    elif language_marker == "javascript":
                        extension = "js"
                    current_block["file_path"] = f"default.{extension}"
                
                file_path_line = ""  # Reset for next block
                
            # End of code block
            elif "```" in line and in_code_block:
                in_code_block = False
                if current_block["file_path"] and current_block["code"].strip():
                    code_blocks.append(current_block)
                current_block = {"file_path": None, "code": ""}
                
            # Content within code block
            elif in_code_block:
                current_block["code"] += line + "\n"
                
        return code_blocks
    
    def extract_file_structure(self, text: str) -> List[str]:
        """
        Extract file structure information from Claude's response.
        
        Args:
            text: Response text from Claude
            
        Returns:
            List of directory paths to create
        """
        directories = []
        lines = text.split('\n')
        
        # Look for directory structure sections
        in_structure_section = False
        
        for i, line in enumerate(lines):
            lower_line = line.lower()
            
            # Check for section headers indicating file structure
            if any(marker in lower_line for marker in ["directory structure", "file structure", "project structure"]):
                in_structure_section = True
                continue
                
            # End of structure section
            if in_structure_section and (line.strip() == "" or "##" in line):
                in_structure_section = False
                
            # Process lines in structure section
            if in_structure_section:
                # Clean up the line
                clean_line = line.strip().replace("└── ", "").replace("├── ", "").replace("│   ", "")
                if "/" in clean_line or "\\" in clean_line:
                    directories.append(clean_line)
                elif "." not in clean_line and clean_line and not clean_line.startswith("-"):
                    # Likely a directory name
                    directories.append(clean_line)
        
        return directories

    def write_file(self, file_path: str, code: str) -> Tuple[bool, str]:
        """
        Write code to a file, creating parent directories as needed.
        
        Args:
            file_path: Relative path to write the file
            code: Code content to write
            
        Returns:
            Tuple of (success, message)
        """
        # Clean up the file path
        file_path = file_path.strip().replace('"', '').replace("'", '')
        
        # Handle absolute paths by converting to relative
        if file_path.startswith("/") or (len(file_path) > 1 and file_path[1] == ":"):
            file_path = Path(file_path).name
        
        # Combine with output directory
        full_path = self.output_dir / file_path
        
        try:
            # Create parent directories if they don't exist
            os.makedirs(full_path.parent, exist_ok=True)
            
            # Check if file already exists and content is different
            if full_path.exists():
                with open(full_path, 'r', encoding='utf-8') as f:
                    existing_content = f.read()
                if existing_content == code:
                    return True, f"File {file_path} already exists with same content, skipping."
            
            # Write the file
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(code)
            
            # Register the file
            self.file_registry[file_path] = f"File created: {len(code)} characters"
            return True, f"Successfully wrote file: {file_path}"
            
        except Exception as e:
            return False, f"Error writing file {file_path}: {str(e)}"

    def create_directories(self, directories: List[str]) -> List[str]:
        """
        Create directories from the extracted structure.
        
        Args:
            directories: List of directory paths to create
            
        Returns:
            List of messages about the directories created
        """
        messages = []
        
        for dir_path in directories:
            # Clean up path
            dir_path = dir_path.strip().replace('"', '').replace("'", '')
            
            # Skip if it's actually a file
            if "." in dir_path.split("/")[-1] or "." in dir_path.split("\\")[-1]:
                continue
                
            # Create the directory
            full_path = self.output_dir / dir_path
            try:
                os.makedirs(full_path, exist_ok=True)
                messages.append(f"Created directory: {dir_path}")
            except Exception as e:
                messages.append(f"Error creating directory {dir_path}: {str(e)}")
                
        return messages

    def process_response(self, response_text: str) -> List[str]:
        """
        Process Claude's response, extracting and creating files and directories.
        
        Args:
            response_text: The response from Claude
            
        Returns:
            List of messages about the processing results
        """
        messages = []
        
        # First, extract and create directory structure
        directories = self.extract_file_structure(response_text)
        if directories:
            dir_messages = self.create_directories(directories)
            messages.extend(dir_messages)
        
        # Then, extract and create code files
        code_blocks = self.parse_code_blocks(response_text)
        for block in code_blocks:
            if block["file_path"] and block["code"]:
                success, msg = self.write_file(block["file_path"], block["code"])
                messages.append(msg)
        
        return messages

    def generate_next_prompt(self) -> str:
        """
        Generate the next prompt to continue code generation.
        
        Returns:
            Prompt for the next API call
        """
        # Get information about files generated so far
        files_info = "\n".join([f"{path}: {info}" for path, info in self.file_registry.items()])
        
        prompt = f"""
I've generated the following files so far:
{files_info}

Please continue the code generation for the project. Focus on:
1. Implementing any missing functionality
2. Improving existing code if needed
3. Adding any necessary files that haven't been created yet
4. Ensuring everything works together cohesively

You can refer to the files I've already created. Please provide the next set of files or updates.
"""
        return prompt

    def start_generation(self, initial_prompt: str) -> None:
        """
        Start the code generation process.
        
        Args:
            initial_prompt: The initial user prompt describing the project
        """
        self.create_directory_structure()
        
        # Add the initial prompt to conversation history
        self.conversation_history.append({"role": "user", "content": initial_prompt})
        
        print("\n🚀 Starting code generation with Claude 3.7...")
        
        # Initial generation
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=self.conversation_history
        )
        
        response_text = response.content[0].text
        print("\n🤖 Claude responded with initial code suggestions.")
        
        # Process the response
        messages = self.process_response(response_text)
        for msg in messages:
            print(f"📁 {msg}")
            
        # Add the response to conversation history
        self.conversation_history.append({"role": "assistant", "content": response_text})
        
        # Continue generation until user stops
        should_continue = True
        generation_round = 1
        
        while should_continue:
            # Generate next prompt
            next_prompt = self.generate_next_prompt()
            
            # Ask user if they want to continue or modify the prompt
            print(f"\n📝 Proposed prompt for round {generation_round + 1}:\n{next_prompt}")
            user_input = input("\nContinue with this prompt? (yes/no/modify): ").strip().lower()
            
            if user_input == "no":
                should_continue = False
                print("\n✅ Code generation completed.")
                continue
            elif user_input == "modify":
                custom_prompt = input("\nEnter custom prompt: ")
                next_prompt = custom_prompt
            
            # Add the next prompt to conversation history
            self.conversation_history.append({"role": "user", "content": next_prompt})
            
            print(f"\n🔄 Generating code for round {generation_round + 1}...")
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    messages=self.conversation_history
                )
                
                response_text = response.content[0].text
                print(f"\n🤖 Claude responded with code for round {generation_round + 1}.")
                
                # Process the response
                messages = self.process_response(response_text)
                for msg in messages:
                    print(f"📁 {msg}")
                    
                # Add the response to conversation history
                self.conversation_history.append({"role": "assistant", "content": response_text})
                
                generation_round += 1
                
            except Exception as e:
                print(f"\n❌ Error in generation round {generation_round + 1}: {str(e)}")
                retry = input("\nRetry this round? (yes/no): ").strip().lower()
                if retry != "yes":
                    should_continue = False
        
        print(f"\n🎉 Project generation complete! Files are available in: {self.output_dir}")
        print(f"Generated {len(self.file_registry)} files across {generation_round} rounds.")

def main():
    """Main function to parse arguments and start generation."""
    parser = argparse.ArgumentParser(description='Generate code projects using Claude 3.7 API')
    parser.add_argument('--api-key', type=str, help='Anthropic API Key (or set ANTHROPIC_API_KEY env var)')
    parser.add_argument('--model', type=str, default="claude-3-7-sonnet-20250219", 
                        help='Claude model to use (default: claude-3-7-sonnet-20250219)')
    parser.add_argument('--output-dir', type=str, default="generated_project", 
                        help='Directory to save generated files')
    parser.add_argument('--max-tokens', type=int, default=4000, 
                        help='Maximum tokens per request (default: 4000)')
    parser.add_argument('--temperature', type=float, default=0.2, 
                        help='Sampling temperature (default: 0.2)')
    parser.add_argument('--prompt-file', type=str, help='File containing the initial prompt')
    
    args = parser.parse_args()
    
    # Get API key from args or environment variable
    api_key = args.api_key or os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        print("Error: Anthropic API key is required. Provide it with --api-key or set ANTHROPIC_API_KEY env var.")
        sys.exit(1)
    
    # Get initial prompt
    if args.prompt_file:
        try:
            with open(args.prompt_file, 'r', encoding='utf-8') as f:
                initial_prompt = f.read()
        except Exception as e:
            print(f"Error reading prompt file: {str(e)}")
            sys.exit(1)
    else:
        print("\n📋 Please enter your initial prompt describing the code project:")
        initial_prompt = ""
        print("(Type 'END' on a new line when finished)")
        while True:
            line = input()
            if line.strip() == "END":
                break
            initial_prompt += line + "\n"
    
    # Create generator and start generation
    generator = ClaudeContinuousCodeGenerator(
        api_key=api_key,
        model=args.model,
        max_tokens_per_request=args.max_tokens,
        temperature=args.temperature,
        output_dir=args.output_dir
    )
    
    generator.start_generation(initial_prompt)

if __name__ == "__main__":
    main()