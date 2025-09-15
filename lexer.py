"""
Lexer for Classes Everywhere Language
Tokenizes source code into a stream of tokens
"""

import re
from enum import Enum, auto
from dataclasses import dataclass
from typing import List, Optional

class TokenType(Enum):
    # Literals
    NUMBER = auto()
    STRING = auto()
    IDENTIFIER = auto()
    CLASS_PATH = auto()  # Thing/A/B
    
    # Keywords
    FUNCTION = auto()
    IF = auto()
    ELSEIF = auto()
    ELSE = auto()
    FOR = auto()
    RETURN = auto()
    NEW = auto()
    ON = auto()
    INT = auto()
    STRING_TYPE = auto()
    FIRE = auto()
    
    # Operators
    ASSIGN = auto()      # =
    PLUS = auto()        # +
    MINUS = auto()       # -
    MULTIPLY = auto()    # *
    DIVIDE = auto()      # /
    MODULO = auto()      # %
    CONCAT = auto()      # &=
    
    # Comparison
    EQUAL = auto()       # ==
    NOT_EQUAL = auto()   # !=
    LESS = auto()        # <
    GREATER = auto()     # >
    LESS_EQUAL = auto()  # <=
    GREATER_EQUAL = auto() # >=
    
    # Delimiters
    LPAREN = auto()      # (
    RPAREN = auto()      # )
    LBRACE = auto()      # {
    RBRACE = auto()      # }
    SEMICOLON = auto()   # ;
    COMMA = auto()       # ,
    DOT = auto()         # .
    DOTDOT = auto()      # ..
    
    # Special
    EOF = auto()
    NEWLINE = auto()

@dataclass
class Token:
    type: TokenType
    value: str
    line: int
    column: int

class Lexer:
    # Pre-define token mappings as class variables to avoid recreating them
    KEYWORDS = {
        'function': TokenType.FUNCTION,
        'if': TokenType.IF,
        'elseif': TokenType.ELSEIF,
        'else': TokenType.ELSE,
        'for': TokenType.FOR,
        'return': TokenType.RETURN,
        'new': TokenType.NEW,
        'on': TokenType.ON,
        'int': TokenType.INT,
        'string': TokenType.STRING_TYPE,
        'fire': TokenType.FIRE,
    }
    
    # Two-character operators lookup table
    TWO_CHAR_TOKENS = {
        '==': TokenType.EQUAL,
        '!=': TokenType.NOT_EQUAL,
        '<=': TokenType.LESS_EQUAL,
        '>=': TokenType.GREATER_EQUAL,
        '&=': TokenType.CONCAT,
        '..': TokenType.DOTDOT
    }
    
    # Single character tokens lookup table
    SINGLE_CHAR_TOKENS = {
        '=': TokenType.ASSIGN,
        '+': TokenType.PLUS,
        '-': TokenType.MINUS,
        '*': TokenType.MULTIPLY,
        '/': TokenType.DIVIDE,
        '%': TokenType.MODULO,
        '<': TokenType.LESS,
        '>': TokenType.GREATER,
        '(': TokenType.LPAREN,
        ')': TokenType.RPAREN,
        '{': TokenType.LBRACE,
        '}': TokenType.RBRACE,
        ';': TokenType.SEMICOLON,
        ',': TokenType.COMMA,
        '.': TokenType.DOT,
        '\n': TokenType.NEWLINE,
    }
    
    def __init__(self, source: str):
        self.source = source
        self.position = 0
        self.line = 1
        self.column = 1
    
    def current_char(self) -> Optional[str]:
        if self.position >= len(self.source):
            return None
        return self.source[self.position]
    
    def peek_char(self, offset: int = 1) -> Optional[str]:
        peek_pos = self.position + offset
        if peek_pos >= len(self.source):
            return None
        return self.source[peek_pos]
    
    def advance(self):
        if self.position < len(self.source) and self.source[self.position] == '\n':
            self.line += 1
            self.column = 1
        else:
            self.column += 1
        self.position += 1
    
    def skip_whitespace(self):
        while self.current_char() and self.current_char() in ' \t\r':
            self.advance()
    
    def skip_comment(self):
        if self.current_char() == '/' and self.peek_char() == '/':
            while self.current_char() and self.current_char() != '\n':
                self.advance()
    
    def read_number(self) -> Token:
        start_line, start_col = self.line, self.column
        value = ''
        
        while self.current_char() and (self.current_char().isdigit() or self.current_char() == '.'):
            value += self.current_char()
            self.advance()
        
        return Token(TokenType.NUMBER, value, start_line, start_col)
    
    def read_string(self) -> Token:
        start_line, start_col = self.line, self.column
        quote_char = self.current_char()
        self.advance()  # Skip opening quote
        
        value = ''
        while self.current_char() and self.current_char() != quote_char:
            if self.current_char() == '\\':
                self.advance()
                if self.current_char() == 'n':
                    value += '\n'
                elif self.current_char() == 't':
                    value += '\t'
                elif self.current_char() == 'r':
                    value += '\r'
                elif self.current_char() == '\\':
                    value += '\\'
                elif self.current_char() == quote_char:
                    value += quote_char
                else:
                    value += self.current_char()
                self.advance()
            else:
                value += self.current_char()
                self.advance()
        
        if self.current_char() == quote_char:
            self.advance()  # Skip closing quote
        
        return Token(TokenType.STRING, value, start_line, start_col)
    
    def read_identifier_or_class_path(self) -> Token:
        start_line, start_col = self.line, self.column
        value = ''
        
        # Read the identifier/class path
        while (self.current_char() and 
               (self.current_char().isalnum() or self.current_char() in '_/')):
            value += self.current_char()
            self.advance()
        
        # Check if it's a class path (contains '/')
        if '/' in value:
            return Token(TokenType.CLASS_PATH, value, start_line, start_col)
        
        # Check if it's a keyword - use class-level KEYWORDS dictionary
        token_type = self.KEYWORDS.get(value, TokenType.IDENTIFIER)
        return Token(token_type, value, start_line, start_col)
    
    def tokenize(self) -> List[Token]:
        tokens = []
        source_length = len(self.source)
        
        # Pre-compile common character sets for faster membership testing
        digits = set('0123456789')
        quotes = set('"\'')
        id_start_chars = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_')
        
        while self.position < source_length:
            self.skip_whitespace()
            
            if self.position >= source_length:
                break
            
            char = self.source[self.position]
            line, col = self.line, self.column
            
            # Skip comments - fast path
            if char == '/' and self.position + 1 < source_length and self.source[self.position + 1] == '/':
                self.skip_comment()
                continue
            
            # Numbers - fast path
            if char in digits:
                tokens.append(self.read_number())
                continue
            
            # Strings - fast path
            if char in quotes:
                tokens.append(self.read_string())
                continue
            
            # Identifiers and class paths - fast path
            if char in id_start_chars:
                tokens.append(self.read_identifier_or_class_path())
                continue
            
            # Two-character operators - optimized with lookup table
            next_pos = self.position + 1
            if next_pos < source_length:
                next_char = self.source[next_pos]
                two_char = char + next_char
                
                # Use the pre-defined class-level lookup table for two-character operators
                if two_char in self.TWO_CHAR_TOKENS:
                    tokens.append(Token(self.TWO_CHAR_TOKENS[two_char], two_char, line, col))
                    self.advance()
                    self.advance()
                    continue
            
            # Use the pre-defined class-level lookup table for single character tokens
            if char in self.SINGLE_CHAR_TOKENS:
                tokens.append(Token(self.SINGLE_CHAR_TOKENS[char], char, line, col))
                self.advance()
                continue
            
            # Newlines
            if char == '\n':
                tokens.append(Token(TokenType.NEWLINE, char, line, col))
                self.advance()
                continue
            
            # Unknown character
            raise SyntaxError(f"Unexpected character '{char}' at line {line}, column {col}")
        
        tokens.append(Token(TokenType.EOF, '', self.line, self.column))
        return tokens