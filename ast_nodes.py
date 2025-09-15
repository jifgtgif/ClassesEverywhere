"""
Abstract Syntax Tree node definitions for Classes Everywhere Language
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Any, Dict
from dataclasses import dataclass

class ASTNode(ABC):
    """Base class for all AST nodes"""
    pass

# ===== EXPRESSIONS =====

class Expression(ASTNode):
    """Base class for all expressions"""
    pass

@dataclass
class LiteralExpression(Expression):
    """Represents literal values (numbers, strings)"""
    value: Any
    type: str  # 'int', 'string', etc.

@dataclass
class IdentifierExpression(Expression):
    """Represents variable/identifier references"""
    name: str

@dataclass
class BinaryExpression(Expression):
    """Represents binary operations (a + b, a == b, etc.)"""
    left: Expression
    operator: str
    right: Expression

@dataclass
class UnaryExpression(Expression):
    """Represents unary operations (-a, !a, etc.)"""
    operator: str
    operand: Expression

@dataclass
class CallExpression(Expression):
    """Represents function calls"""
    name: str
    arguments: List[Expression]

@dataclass
class MemberExpression(Expression):
    """Represents member access (obj.method)"""
    object: Expression
    property: str

@dataclass
class NewExpression(Expression):
    """Represents object instantiation new("ClassName")"""
    class_name: str

@dataclass
class MethodCallExpression(Expression):
    """Represents method calls (obj.method(args))"""
    object: Expression
    method_name: str
    arguments: List[Expression]

@dataclass
class SuperCallExpression(Expression):
    """Represents super calls (..())"""
    arguments: List[Expression]

# ===== STATEMENTS =====

class Statement(ASTNode):
    """Base class for all statements"""
    pass

@dataclass
class ExpressionStatement(Statement):
    """Wraps an expression as a statement"""
    expression: Expression

@dataclass
class VariableDeclaration(Statement):
    """Represents variable declarations (int x = 5;)"""
    type: str  # 'int', 'string'
    name: str
    initializer: Optional[Expression] = None

@dataclass
class AssignmentStatement(Statement):
    """Represents assignment (x = 5; result &= "text";)"""
    name: str
    operator: str  # '=', '&='
    value: Expression

@dataclass
class MemberAssignmentStatement(Statement):
    """Represents member assignment (obj.field = value)"""
    object: Expression
    property: str
    operator: str  # '=', '&='
    value: Expression

@dataclass
class IfStatement(Statement):
    """Represents if/elseif/else statements"""
    condition: Expression
    then_block: List[Statement]
    elseif_clauses: List['ElseIfClause']
    else_block: Optional[List[Statement]] = None

@dataclass
class ElseIfClause:
    """Represents an elseif clause"""
    condition: Expression
    block: List[Statement]

@dataclass
class ForStatement(Statement):
    """Represents for loops"""
    variable: str
    start: Expression
    condition: Expression
    increment: Optional[Expression]
    body: List[Statement]

@dataclass
class ReturnStatement(Statement):
    """Represents return statements"""
    value: Optional[Expression] = None

@dataclass
class BlockStatement(Statement):
    """Represents a block of statements"""
    statements: List[Statement]

# ===== DECLARATIONS =====

@dataclass
class Parameter:
    """Represents a function parameter"""
    name: str
    type: Optional[str] = None

@dataclass
class FunctionDeclaration(ASTNode):
    """Represents function declarations"""
    name: str
    parameters: List[Parameter]
    body: List[Statement]
    return_type: Optional[str] = None

@dataclass
class ClassDeclaration(ASTNode):
    """Represents class declarations"""
    name: str  # Full path like "Thing/A/B"
    members: List[ASTNode]  # Variables and functions
    parent: Optional[str] = None  # Parent class path

@dataclass
class EventHandler(ASTNode):
    """Represents event handlers (on Events.start)"""
    event_name: str
    body: List[Statement]

# ===== PROGRAM =====

@dataclass
class Program(ASTNode):
    """Represents the entire program"""
    declarations: List[ASTNode]  # Classes, event handlers, etc.

# ===== UTILITY FUNCTIONS =====

def print_ast(node: ASTNode, indent: int = 0) -> str:
    """Pretty print AST for debugging"""
    spaces = "  " * indent
    
    if isinstance(node, Program):
        result = f"{spaces}Program:\n"
        for decl in node.declarations:
            result += print_ast(decl, indent + 1)
        return result
    
    elif isinstance(node, ClassDeclaration):
        result = f"{spaces}Class '{node.name}':\n"
        for member in node.members:
            result += print_ast(member, indent + 1)
        return result
    
    elif isinstance(node, FunctionDeclaration):
        params = ", ".join(p.name for p in node.parameters)
        result = f"{spaces}Function '{node.name}({params})':\n"
        for stmt in node.body:
            result += print_ast(stmt, indent + 1)
        return result
    
    elif isinstance(node, VariableDeclaration):
        init = f" = {print_ast(node.initializer, 0).strip()}" if node.initializer else ""
        return f"{spaces}{node.type} {node.name}{init};\n"
    
    elif isinstance(node, BinaryExpression):
        left = print_ast(node.left, 0).strip()
        right = print_ast(node.right, 0).strip()
        return f"({left} {node.operator} {right})"
    
    elif isinstance(node, LiteralExpression):
        if node.type == 'string':
            return f'"{node.value}"'
        return str(node.value)
    
    elif isinstance(node, IdentifierExpression):
        return node.name
    
    elif isinstance(node, EventHandler):
        result = f"{spaces}Event '{node.event_name}':\n"
        for stmt in node.body:
            result += print_ast(stmt, indent + 1)
        return result
    
    else:
        return f"{spaces}{type(node).__name__}\n"