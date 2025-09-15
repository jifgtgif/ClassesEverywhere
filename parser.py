"""
Recursive Descent Parser for Classes Everywhere Language
Converts tokens into an Abstract Syntax Tree (AST)
"""

from typing import List, Optional, Union
from lexer import Token, TokenType
from ast_nodes import *

class ParseError(Exception):
    """Exception raised when parsing fails"""
    pass

class Parser:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.current = 0
        # Cache for memoization to avoid redundant parsing
        self._expression_cache = {}
        self._statement_cache = {}
    
    def is_at_end(self) -> bool:
        """Check if we've reached the end of tokens"""
        return self.peek().type == TokenType.EOF
    
    def peek(self) -> Token:
        """Return current token without advancing"""
        return self.tokens[self.current]
    
    def previous(self) -> Token:
        """Return previous token"""
        return self.tokens[self.current - 1]
    
    def advance(self) -> Token:
        """Consume and return current token"""
        if not self.is_at_end():
            self.current += 1
        return self.previous()
    
    def check(self, type: TokenType) -> bool:
        """Check if current token is of given type"""
        if self.is_at_end():
            return False
        return self.peek().type == type
    
    def match(self, *types: TokenType) -> bool:
        """Check if current token matches any of the given types"""
        for token_type in types:
            if self.check(token_type):
                self.advance()
                return True
        return False
    
    def consume(self, type: TokenType, message: str) -> Token:
        """Consume token of expected type or raise error"""
        if self.check(type):
            return self.advance()
        
        current_token = self.peek()
        raise ParseError(f"{message}. Got {current_token.type.name} at line {current_token.line}")
    
    def skip_newlines(self):
        """Skip any newline tokens"""
        while self.match(TokenType.NEWLINE):
            pass
    
    def parse(self) -> Program:
        """Parse the entire program"""
        # Clear caches at the start of parsing
        self.clear_caches()
        
        declarations = []
        
        while not self.is_at_end():
            self.skip_newlines()
            if self.is_at_end():
                break
            
            decl = self.declaration()
            if decl:
                declarations.append(decl)
        
        return Program(declarations)
    
    def declaration(self) -> Optional[ASTNode]:
        """Parse top-level declarations"""
        try:
            if self.check(TokenType.CLASS_PATH):
                return self.class_declaration()
            elif self.match(TokenType.ON):
                return self.event_handler()
            else:
                # Skip unexpected tokens at top level
                self.advance()
                return None
        except ParseError as e:
            # Synchronize and continue parsing
            self.synchronize()
            return None
    
    def class_declaration(self) -> ClassDeclaration:
        """Parse class declaration: Thing/A/B = { ... }"""
        class_path = self.advance().value  # CLASS_PATH token
        
        self.consume(TokenType.ASSIGN, "Expected '=' after class name")
        self.consume(TokenType.LBRACE, "Expected '{' after '='")
        
        self.skip_newlines()
        
        members = []
        while not self.check(TokenType.RBRACE) and not self.is_at_end():
            self.skip_newlines()
            if self.check(TokenType.RBRACE):
                break
                
            member = self.class_member()
            if member:
                members.append(member)
            
            self.skip_newlines()
        
        self.consume(TokenType.RBRACE, "Expected '}' after class body")
        
        # Determine parent class from path
        # Only classes with 3+ segments inherit (A/B/C inherits from A/B)
        parent = None
        if '/' in class_path:
            parts = class_path.split('/')
            if len(parts) >= 3:  # A/B/C inherits from A/B, but A/B is standalone
                parent = '/'.join(parts[:-1])
        
        return ClassDeclaration(name=class_path, members=members, parent=parent)
    
    def class_member(self) -> Optional[ASTNode]:
        """Parse class member (variable or function)"""
        if self.match(TokenType.INT, TokenType.STRING_TYPE):
            type_name = self.previous().value
            return self.variable_declaration(type_name)
        elif self.match(TokenType.FUNCTION):
            return self.function_declaration()
        else:
            # Skip unexpected tokens
            self.advance()
            return None
    
    def variable_declaration(self, type_name: str) -> VariableDeclaration:
        """Parse variable declaration: int x = 10;"""
        name = self.consume(TokenType.IDENTIFIER, "Expected variable name").value
        
        initializer = None
        if self.match(TokenType.ASSIGN):
            initializer = self.expression()
        
        self.consume(TokenType.SEMICOLON, "Expected ';' after variable declaration")
        
        return VariableDeclaration(type=type_name, name=name, initializer=initializer)
    
    def function_declaration(self) -> FunctionDeclaration:
        """Parse function declaration: function name() { ... }"""
        # Allow 'new' and 'fire' as function names
        if self.match(TokenType.NEW):
            name = "new"
        elif self.match(TokenType.FIRE):
            name = "Fire"
        else:
            name = self.consume(TokenType.IDENTIFIER, "Expected function name").value
        
        self.consume(TokenType.LPAREN, "Expected '(' after function name")
        
        parameters = []
        if not self.check(TokenType.RPAREN):
            parameters.append(Parameter(name=self.consume(TokenType.IDENTIFIER, "Expected parameter name").value))
            
            while self.match(TokenType.COMMA):
                parameters.append(Parameter(name=self.consume(TokenType.IDENTIFIER, "Expected parameter name").value))
        
        self.consume(TokenType.RPAREN, "Expected ')' after parameters")
        
        self.consume(TokenType.LBRACE, "Expected '{' before function body")
        
        body = self.block()
        
        return FunctionDeclaration(name=name, parameters=parameters, body=body)
    
    def event_handler(self) -> EventHandler:
        """Parse event handler: on Events.start or on Thing/Event/EventName { ... }"""
        # Check if it's a class path (Thing/Event/EventName) or regular event (Events.start)
        if self.check(TokenType.CLASS_PATH):
            # Custom event: Thing/Event/EventName
            event_name = self.advance().value
        else:
            # Built-in event: Events.start or Events.draw
            event_path = []
            event_path.append(self.consume(TokenType.IDENTIFIER, "Expected event category").value)
            
            self.consume(TokenType.DOT, "Expected '.' in event name")
            event_path.append(self.consume(TokenType.IDENTIFIER, "Expected event name").value)
            
            event_name = '.'.join(event_path)
        
        self.consume(TokenType.LBRACE, "Expected '{' after event name")
        
        body = self.block()
        
        return EventHandler(event_name=event_name, body=body)
    
    def block(self) -> List[Statement]:
        """Parse block of statements"""
        statements = []
        
        self.skip_newlines()
        
        while not self.check(TokenType.RBRACE) and not self.is_at_end():
            self.skip_newlines()
            if self.check(TokenType.RBRACE):
                break
                
            stmt = self.statement()
            if stmt:
                statements.append(stmt)
            
            self.skip_newlines()
        
        self.consume(TokenType.RBRACE, "Expected '}' after block")
        
        return statements
    
    def statement(self) -> Optional[Statement]:
        """Parse individual statement with memoization"""
        # Use current position as cache key
        cache_key = self.current
        
        # Check if we've already parsed a statement at this position
        if cache_key in self._statement_cache:
            # Restore the position after the parsed statement
            result, new_position = self._statement_cache[cache_key]
            self.current = new_position
            return result
            
        # Parse the statement
        start_pos = self.current
        try:
            if self.match(TokenType.IF):
                result = self.if_statement()
            elif self.match(TokenType.FOR):
                result = self.for_statement()
            elif self.match(TokenType.RETURN):
                result = self.return_statement()
            else:
                result = self.expression_statement()
                
            # Cache the result and the new position
            self._statement_cache[start_pos] = (result, self.current)
            return result
        except ParseError as e:
            self.synchronize()
            return None
    
    def if_statement(self) -> IfStatement:
        """Parse if statement"""
        self.consume(TokenType.LPAREN, "Expected '(' after 'if'")
        condition = self.expression()
        self.consume(TokenType.RPAREN, "Expected ')' after if condition")
        
        self.consume(TokenType.LBRACE, "Expected '{' after if condition")
        then_block = self.block()
        
        elseif_clauses = []
        while self.match(TokenType.ELSEIF):
            self.consume(TokenType.LPAREN, "Expected '(' after 'elseif'")
            elseif_condition = self.expression()
            self.consume(TokenType.RPAREN, "Expected ')' after elseif condition")
            
            self.consume(TokenType.LBRACE, "Expected '{' after elseif condition")
            elseif_block = self.block()
            
            elseif_clauses.append(ElseIfClause(condition=elseif_condition, block=elseif_block))
        
        else_block = None
        if self.match(TokenType.ELSE):
            self.consume(TokenType.LBRACE, "Expected '{' after 'else'")
            else_block = self.block()
        
        return IfStatement(condition=condition, then_block=then_block, 
                          elseif_clauses=elseif_clauses, else_block=else_block)
    
    def for_statement(self) -> ForStatement:
        """Parse for statement: for (i=0;i<10;i++)"""
        self.consume(TokenType.LPAREN, "Expected '(' after 'for'")
        
        # Parse initialization (i=0)
        var_name = self.consume(TokenType.IDENTIFIER, "Expected variable name").value
        self.consume(TokenType.ASSIGN, "Expected '=' in for loop initialization")
        start = self.expression()
        self.consume(TokenType.SEMICOLON, "Expected ';' after for loop initialization")
        
        # Parse condition (i<10)
        condition = self.expression()
        self.consume(TokenType.SEMICOLON, "Expected ';' after for loop condition")
        
        # Parse increment (i++)
        increment = None
        if not self.check(TokenType.RPAREN):
            # Handle i++ as i = i + 1
            if self.check(TokenType.IDENTIFIER):
                var_token = self.advance()
                if (self.check(TokenType.PLUS) and 
                    self.current + 1 < len(self.tokens) and 
                    self.tokens[self.current + 1].type == TokenType.PLUS):  # i++
                    self.advance()  # consume first +
                    self.advance()  # consume second +
                    # Create i = i + 1 expression
                    increment = BinaryExpression(
                        left=IdentifierExpression(name=var_token.value),
                        operator="=",
                        right=BinaryExpression(
                            left=IdentifierExpression(name=var_token.value),
                            operator="+",
                            right=LiteralExpression(value=1, type='int')
                        )
                    )
                else:
                    # Backtrack and parse as regular expression
                    self.current -= 1
                    increment = self.expression()
            else:
                increment = self.expression()
        
        self.consume(TokenType.RPAREN, "Expected ')' after for loop")
        
        self.consume(TokenType.LBRACE, "Expected '{' after for loop")
        body = self.block()
        
        return ForStatement(variable=var_name, start=start, condition=condition, 
                          increment=increment, body=body)
    
    def return_statement(self) -> ReturnStatement:
        """Parse return statement"""
        value = None
        if not self.check(TokenType.SEMICOLON):
            value = self.expression()
        
        self.consume(TokenType.SEMICOLON, "Expected ';' after return value")
        
        return ReturnStatement(value=value)
    
    def expression_statement(self) -> Statement:
        """Parse expression statement (assignment or function call)"""
        expr = self.expression()
        self.consume(TokenType.SEMICOLON, "Expected ';' after expression")
        
        # Convert BinaryExpression assignments to AssignmentStatement or MemberAssignmentStatement
        if isinstance(expr, BinaryExpression) and expr.operator in ["=", "&="]:
            if isinstance(expr.left, IdentifierExpression):
                return AssignmentStatement(
                    name=expr.left.name,
                    operator=expr.operator,
                    value=expr.right
                )
            elif isinstance(expr.left, MemberExpression):
                return MemberAssignmentStatement(
                    object=expr.left.object,
                    property=expr.left.property,
                    operator=expr.operator,
                    value=expr.right
                )
        
        return ExpressionStatement(expression=expr)
    
    def expression(self) -> Expression:
        """Parse expression with memoization"""
        # Use current position as cache key
        cache_key = self.current
        
        # Check if we've already parsed an expression at this position
        if cache_key in self._expression_cache:
            # Restore the position after the parsed expression
            result, new_position = self._expression_cache[cache_key]
            self.current = new_position
            return result
            
        # Parse the expression
        start_pos = self.current
        result = self.assignment()
        
        # Cache the result and the new position
        self._expression_cache[start_pos] = (result, self.current)
        
        return result
    
    def assignment(self) -> Expression:
        """Parse assignment expression"""
        expr = self.logical_or()
        
        if self.match(TokenType.ASSIGN, TokenType.CONCAT):
            operator = self.previous().value
            value = self.assignment()
            
            if isinstance(expr, IdentifierExpression) or isinstance(expr, MemberExpression):
                return BinaryExpression(left=expr, operator=operator, right=value)
            
            raise ParseError("Invalid assignment target")
        
        return expr
    
    def logical_or(self) -> Expression:
        """Parse logical OR expression"""
        expr = self.logical_and()
        
        while self.match(TokenType.EQUAL):  # Using == as OR for now
            operator = self.previous().value
            right = self.logical_and()
            expr = BinaryExpression(left=expr, operator=operator, right=right)
        
        return expr
    
    def logical_and(self) -> Expression:
        """Parse logical AND expression"""
        expr = self.equality()
        
        # Add logical AND support if needed
        return expr
    
    def equality(self) -> Expression:
        """Parse equality expression"""
        expr = self.comparison()
        
        while self.match(TokenType.EQUAL, TokenType.NOT_EQUAL):
            operator = self.previous().value
            right = self.comparison()
            expr = BinaryExpression(left=expr, operator=operator, right=right)
        
        return expr
    
    def comparison(self) -> Expression:
        """Parse comparison expression"""
        expr = self.term()
        
        while self.match(TokenType.GREATER, TokenType.GREATER_EQUAL, TokenType.LESS, TokenType.LESS_EQUAL):
            operator = self.previous().value
            right = self.term()
            expr = BinaryExpression(left=expr, operator=operator, right=right)
        
        return expr
    
    def term(self) -> Expression:
        """Parse addition/subtraction expression"""
        expr = self.factor()
        
        while self.match(TokenType.MINUS, TokenType.PLUS):
            operator = self.previous().value
            right = self.factor()
            expr = BinaryExpression(left=expr, operator=operator, right=right)
        
        return expr
    
    def factor(self) -> Expression:
        """Parse multiplication/division/modulo expression"""
        expr = self.unary()
        
        while self.match(TokenType.DIVIDE, TokenType.MULTIPLY, TokenType.MODULO):
            operator = self.previous().value
            right = self.unary()
            expr = BinaryExpression(left=expr, operator=operator, right=right)
        
        return expr
    
    def unary(self) -> Expression:
        """Parse unary expression"""
        if self.match(TokenType.MINUS):
            operator = self.previous().value
            right = self.unary()
            return UnaryExpression(operator=operator, operand=right)
        
        return self.call()
    
    def call(self) -> Expression:
        """Parse function call or member access"""
        expr = self.primary()
        
        while True:
            if self.match(TokenType.LPAREN):
                expr = self.finish_call(expr)
            elif self.match(TokenType.DOT):
                name = self.consume(TokenType.IDENTIFIER, "Expected property name after '.'").value
                expr = MemberExpression(object=expr, property=name)
            else:
                break
        
        return expr
    
    def finish_call(self, callee: Expression) -> Expression:
        """Parse function call arguments"""
        arguments = []
        
        if not self.check(TokenType.RPAREN):
            arguments.append(self.expression())
            while self.match(TokenType.COMMA):
                arguments.append(self.expression())
        
        self.consume(TokenType.RPAREN, "Expected ')' after arguments")
        
        if isinstance(callee, IdentifierExpression):
            return CallExpression(name=callee.name, arguments=arguments)
        elif isinstance(callee, MemberExpression):
            # Handle method calls
            return MethodCallExpression(object=callee.object, method_name=callee.property, arguments=arguments)
        else:
            raise ParseError("Invalid function call")
    
    def primary(self) -> Expression:
        """Parse primary expression"""
        if self.match(TokenType.NUMBER):
            value = self.previous().value
            if '.' in value:
                return LiteralExpression(value=float(value), type='float')
            else:
                return LiteralExpression(value=int(value), type='int')
        
        if self.match(TokenType.STRING):
            return LiteralExpression(value=self.previous().value, type='string')
        
        if self.match(TokenType.IDENTIFIER):
            return IdentifierExpression(name=self.previous().value)
        
        if self.match(TokenType.NEW):
            self.consume(TokenType.LPAREN, "Expected '(' after 'new'")
            class_name = self.consume(TokenType.STRING, "Expected class name string").value
            self.consume(TokenType.RPAREN, "Expected ')' after class name")
            return NewExpression(class_name=class_name)
        
        if self.match(TokenType.LPAREN):
            expr = self.expression()
            self.consume(TokenType.RPAREN, "Expected ')' after expression")
            return expr
        
        if self.match(TokenType.DOTDOT):
            # Super call: ..(args)
            self.consume(TokenType.LPAREN, "Expected '(' after '..'")
            arguments = []
            if not self.check(TokenType.RPAREN):
                arguments.append(self.expression())
                while self.match(TokenType.COMMA):
                    arguments.append(self.expression())
            self.consume(TokenType.RPAREN, "Expected ')' after super call arguments")
            return SuperCallExpression(arguments=arguments)
        
        raise ParseError(f"Unexpected token {self.peek().type.name}")
    
    def clear_caches(self):
        """Clear the memoization caches"""
        self._expression_cache.clear()
        self._statement_cache.clear()
    
    def synchronize(self):
        """Synchronize after parse error"""
        self.advance()
        
        # Clear caches when synchronizing to avoid using stale cached values
        self.clear_caches()
        
        while not self.is_at_end():
            if self.previous().type == TokenType.SEMICOLON:
                return
            
            if self.peek().type in [TokenType.CLASS_PATH, TokenType.FUNCTION, TokenType.IF, 
                                   TokenType.FOR, TokenType.RETURN, TokenType.ON]:
                return
            
            self.advance()