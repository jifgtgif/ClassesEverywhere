#!/usr/bin/env python3
"""
Classes Everywhere Language Interpreter
Main entry point for the language implementation
"""

import sys
from lexer import Lexer
from parser import Parser
from interpreter import Interpreter

def main():
    if len(sys.argv) != 2:
        print("Usage: python main.py <filename>")
        sys.exit(1)
    
    filename = sys.argv[1]
    
    try:
        with open(filename, 'r') as file:
            source_code = file.read()
        
        # Tokenize the source code
        lexer = Lexer(source_code)
        tokens = lexer.tokenize()
        
        # Parse tokens into AST
        parser = Parser(tokens)
        ast = parser.parse()
        
        # Execute the program
        interpreter = Interpreter()
        interpreter.execute(ast)
        
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()