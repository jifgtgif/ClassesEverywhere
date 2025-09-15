"""
Interpreter for Classes Everywhere Language
Includes symbol table, scope management, and execution engine
"""
from math import *
import pygame
from typing import Any, Dict, List, Optional, Union
from ast_nodes import *

class RuntimeError(Exception):
    """Exception raised during runtime execution"""
    pass

class ClassInstance:
    """Represents an instance of a class"""
    def __init__(self, class_def: 'ClassDefinition'):
        self.class_def = class_def
        self.fields = {}
        
    def get(self, name: str) -> Any:
        """Get field or method from instance"""
        # First check if the field exists in the instance
        if name in self.fields:
            return self.fields[name]
            
        # Check if it's a field defined in the class but not yet in the instance
        try:
            if self.class_def.has_field(name):
                # Get the field's initial value from the class definition
                value = self.class_def.get_field(name)
                # Store it in the instance for future access
                self.fields[name] = value
                return value
        except Exception as e:
            # If there's an error accessing the field, provide a clear error message
            raise RuntimeError(f"Error accessing field '{name}': {e}")
            
        # If it's not a field, check if it's a method
        method = self.class_def.get_method(name)
        if method is not None:
            return method
            
        # If we get here, the field or method doesn't exist
        raise RuntimeError(f"Undefined field or method '{name}' in class '{self.class_def.name}'")

    
    def set(self, name: str, value: Any):
        """Set field value"""
        self.fields[name] = value
    
    def has_field(self, name: str) -> bool:
        """Check if field exists"""
        return name in self.fields or self.class_def.has_field(name)

class ClassDefinition:
    """Represents a class definition with inheritance"""
    def __init__(self, name: str, parent: Optional['ClassDefinition'] = None):
        self.name = name
        self.parent = parent
        self.fields = {}  # name -> initial value
        self.methods = {}  # name -> FunctionDeclaration
        # Caches for method and field lookups
        self._method_cache = {}
        self._field_cache = {}
        self._field_exists_cache = {}
        
    def add_field(self, name: str, type_name: str, initial_value: Any = None):
        """Add a field to the class"""
        # Invalidate caches for this field
        if name in self._field_cache:
            del self._field_cache[name]
        if name in self._field_exists_cache:
            del self._field_exists_cache[name]
            
        self.fields[name] = initial_value
    
    def add_method(self, name: str, method: FunctionDeclaration):
        """Add a method to the class"""
        # Invalidate method cache
        if name in self._method_cache:
            del self._method_cache[name]
            
        self.methods[name] = method
    
    def get_field(self, name: str) -> Any:
        """Get field value (including from parent classes)"""
        # Check cache first
        if name in self._field_cache:
            field_value, exists = self._field_cache[name]
            if exists:
                return field_value
            raise RuntimeError(f"Undefined field '{name}' in class '{self.name}'")
            
        # Check current class
        if name in self.fields:
            self._field_cache[name] = (self.fields[name], True)
            return self.fields[name]
            
        # Check parent classes
        if self.parent:
            try:
                value = self.parent.get_field(name)
                self._field_cache[name] = (value, True)
                return value
            except RuntimeError:
                self._field_cache[name] = (None, False)
                raise
                
        # Field not found
        self._field_cache[name] = (None, False)
        raise RuntimeError(f"Undefined field '{name}' in class '{self.name}'")

    
    def get_method(self, name: str) -> Optional[FunctionDeclaration]:
        """Get method (including from parent classes)"""
        # Check cache first
        if name in self._method_cache:
            return self._method_cache[name]
            
        # Check current class
        if name in self.methods:
            self._method_cache[name] = self.methods[name]
            return self.methods[name]
            
        # Check parent classes
        if self.parent:
            method = self.parent.get_method(name)
            self._method_cache[name] = method
            return method
            
        # Method not found
        self._method_cache[name] = None
        return None
    
    def get_parent_method(self, name: str) -> Optional[FunctionDeclaration]:
        """Get method from parent class only (for super calls)"""
        if self.parent:
            return self.parent.get_method(name)
        return None
    
    def has_field(self, name: str) -> bool:
        """Check if field exists (including in parent classes)"""
        # Check cache first
        if name in self._field_exists_cache:
            return self._field_exists_cache[name]
            
        # Check current class
        if name in self.fields:
            self._field_exists_cache[name] = True
            return True
            
        # Check parent classes
        if self.parent:
            exists = self.parent.has_field(name)
            self._field_exists_cache[name] = exists
            return exists
            
        # Field not found
        self._field_exists_cache[name] = False
        return False
    
    def instantiate(self) -> ClassInstance:
        """Create a new instance of this class"""
        instance = ClassInstance(self)
        
        # Initialize fields from this class and all parent classes
        def init_fields(class_def: ClassDefinition):
            if class_def.parent:
                init_fields(class_def.parent)
            for name, value in class_def.fields.items():
                instance.fields[name] = value
        
        init_fields(self)
        return instance

class Environment:
    """Represents a scope for variable bindings"""
    def __init__(self, enclosing: Optional['Environment'] = None):
        self.enclosing = enclosing
        self.values = {}
        # Cache for variable lookups to avoid traversing the environment chain repeatedly
        self._lookup_cache = {}
    
    def define(self, name: str, value: Any):
        """Define a variable in this scope"""
        # Update cache when defining a new variable
        self._lookup_cache[name] = (self, True)
        self.values[name] = value
    
    def get(self, name: str) -> Any:
        """Get variable value from this scope or enclosing scopes"""
        # Check cache first
        if name in self._lookup_cache:
            env, is_defined = self._lookup_cache[name]
            if is_defined:
                return env.values[name]
            raise RuntimeError(f"Undefined variable '{name}'")
            
        # Check current scope
        if name in self.values:
            self._lookup_cache[name] = (self, True)
            return self.values[name]
        
        # Check enclosing scopes
        if self.enclosing:
            try:
                value = self.enclosing.get(name)
                # Cache the successful lookup
                self._lookup_cache[name] = (self.enclosing, True)
                return value
            except RuntimeError:
                # Cache the failed lookup
                self._lookup_cache[name] = (None, False)
                raise
            except KeyError:
                # Handle KeyError separately to provide better error message
                self._lookup_cache[name] = (None, False)
                raise RuntimeError(f"Undefined variable '{name}'")
        
        # Not found anywhere
        self._lookup_cache[name] = (None, False)
        raise RuntimeError(f"Undefined variable '{name}'")

    
    def assign(self, name: str, value: Any):
        """Assign to a variable (create if doesn't exist)"""
        # Invalidate cache for this name since we're modifying it
        if name in self._lookup_cache:
            del self._lookup_cache[name]
            
        if name in self.values:
            self.values[name] = value
            return
        
        if self.enclosing:
            try:
                self.enclosing.assign(name, value)
                return
            except RuntimeError:
                # Variable doesn't exist in enclosing scopes, create it here
                pass
        
        # Define the variable in current scope if it doesn't exist anywhere
        self.values[name] = value

class Function:
    """Represents a callable function"""
    def __init__(self, declaration: FunctionDeclaration, closure: Environment, instance: Optional[ClassInstance] = None):
        self.declaration = declaration
        self.closure = closure
        self.instance = instance  # For methods
        self.current_method_name = declaration.name if instance else None
    
    def call(self, interpreter: 'Interpreter', arguments: List[Any]) -> Any:
        """Call this function with given arguments"""
        if len(arguments) != len(self.declaration.parameters):
            raise RuntimeError(f"Expected {len(self.declaration.parameters)} arguments but got {len(arguments)}")
        
        # Create new environment for function execution
        environment = Environment(self.closure)
        
        # Bind parameters
        for i, param in enumerate(self.declaration.parameters):
            environment.define(param.name, arguments[i])
        
        # Bind 'this' if this is a method call
        if self.instance:
            environment.define("this", self.instance)
            # Store current method context for super calls
            environment.define("__current_method__", self.current_method_name)
            environment.define("__current_instance__", self.instance)
            # Also bind instance fields directly (create live references)
            for name, value in self.instance.fields.items():
                environment.define(name, value)
            
            # Override field assignment to update instance fields
            original_assign = environment.assign
            def field_aware_assign(name: str, value: Any):
                if name in self.instance.fields:
                    self.instance.set(name, value)
                    environment.values[name] = value  # Keep environment in sync
                else:
                    original_assign(name, value)
            environment.assign = field_aware_assign
            # Bind instance methods as callable functions
            for name, method in self.instance.class_def.methods.items():
                if name != self.declaration.name:  # Don't bind self
                    environment.define(name, Function(method, environment, self.instance))
        
        try:
            interpreter.execute_block(self.declaration.body, environment)
        except ReturnValue as return_val:
            return return_val.value
        
        return None

class ReturnValue(Exception):
    """Exception used to implement return statements"""
    def __init__(self, value: Any):
        self.value = value

class Interpreter:
    def __init__(self):
        self.globals = Environment()
        self.environment = self.globals
        self.classes = {}  # name -> ClassDefinition
        self.events = {}  # name -> EventHandler
        
        pygame.init()
        self.width, self.height = 640, 400   # Window size (scalable)
        self.pixel_size = 1                  # Each "pixel" in logical canvas
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Classes Everywhere Language")
        self.clock = pygame.time.Clock()

        # Logical canvas resolution (like old text mode)
        self.canvas_width = self.width // self.pixel_size
        self.canvas_height = self.height // self.pixel_size
        self.canvas = [[(0,0,0) for _ in range(self.canvas_width)] for _ in range(self.canvas_height)]
        self.current_color = (255,255,255)

        
        # Define built-in functions
        self.define_builtins()
    
    def define_builtins(self):
        """Define built-in functions like print"""
        def builtin_print(*args):
            print(*args)
            return None
        
        def builtin_set_color(r, g, b):
            """Set the current drawing color"""
            self.current_color = (int(r), int(g), int(b))
            return None
        
        def builtin_draw_pixel(x, y):
            x, y = int(x), int(y)
            if 0 <= x < self.canvas_width and 0 <= y < self.canvas_height:
                self.canvas[y][x] = self.current_color
            return None

        def builtin_clear_canvas():
            self.canvas = [[(0,0,0) for _ in range(self.canvas_width)] for _ in range(self.canvas_height)]
            return None
        
        def builtin_get_mousepos_x():
            return pygame.mouse.get_pos()[0]
        
        def builtin_get_mousepos_y():
            return pygame.mouse.get_pos()[1]

        def builtin_show_canvas():
            # Draw each logical pixel as a square
            for y in range(self.canvas_height):
                for x in range(self.canvas_width):
                    color = self.canvas[y][x]
                    rect = pygame.Rect(x*self.pixel_size, y*self.pixel_size, self.pixel_size, self.pixel_size)
                    pygame.draw.rect(self.screen, color, rect)
            pygame.display.flip()
            return None

        
        def builtin_fire(event_name):
            """Fire a custom event"""
            if event_name in self.events:
                self.execute_block(self.events[event_name].body, self.environment)
            return None
        
        self.globals.define("print", builtin_print)
        self.globals.define("SetColor", builtin_set_color)
        self.globals.define("DrawPixel", builtin_draw_pixel)
        self.globals.define("ClearCanvas", builtin_clear_canvas)
        self.globals.define("ShowCanvas", builtin_show_canvas)
        self.globals.define("Fire", builtin_fire)
        self.globals.define("sin", sin)
        self.globals.define("cos", cos)
        self.globals.define("tan", tan)
        self.globals.define("radians", radians)
        self.globals.define("tan", tan)
        self.globals.define("mousex", builtin_get_mousepos_x)
        self.globals.define("mousey", builtin_get_mousepos_y)
        
    
    def execute(self, program: Program):
        """Execute the entire program"""
        try:
            # First pass: collect all class declarations
            class_declarations = []
            for declaration in program.declarations:
                if isinstance(declaration, ClassDeclaration):
                    class_declarations.append(declaration)
            
            # Sort class declarations by inheritance hierarchy
            # (parents before children)
            defined_classes = set()
            
            while class_declarations:
                progress_made = False
                for i, class_decl in enumerate(class_declarations):
                    # Check if parent is already defined (or no parent)
                    if not class_decl.parent or class_decl.parent in defined_classes:
                        self.define_class(class_decl)
                        defined_classes.add(class_decl.name)
                        class_declarations.pop(i)
                        progress_made = True
                        break
                
                if not progress_made:
                    # Circular dependency or missing parent
                    remaining = [c.name for c in class_declarations]
                    remaining_parents = [(c.name, c.parent) for c in class_declarations]
                    raise RuntimeError(f"Cannot resolve class dependencies: {remaining_parents}")
            
            # Store event handlers
            for declaration in program.declarations:
                if isinstance(declaration, EventHandler):
                    self.events[declaration.event_name] = declaration

            # Run Events.start
            if "Events.start" in self.events:
                self.execute_block(self.events["Events.start"].body, self.environment)

            running = True
            while running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False

                # Fire Events.draw every frame
                if "Events.draw" in self.events:
                    self.execute_block(self.events["Events.draw"].body, self.environment)

                self.clock.tick(30)  # Limit to 30 FPS

        except RuntimeError as e:
            print(f"Runtime error: {e}")
        finally:
            pygame.quit()

    
    def define_class(self, class_decl: ClassDeclaration):
        """Define a class and add it to the class table"""
        # Get parent class if it exists
        parent_class = None
        if class_decl.parent:
            if class_decl.parent in self.classes:
                parent_class = self.classes[class_decl.parent]
            else:
                raise RuntimeError(f"Undefined parent class '{class_decl.parent}'")
        
        # Create class definition
        class_def = ClassDefinition(class_decl.name, parent_class)
        
        # Add fields and methods
        for member in class_decl.members:
            if isinstance(member, VariableDeclaration):
                initial_value = None
                if member.initializer:
                    initial_value = self.evaluate(member.initializer)
                class_def.add_field(member.name, member.type, initial_value)
            
            elif isinstance(member, FunctionDeclaration):
                class_def.add_method(member.name, member)
        
        # Store class definition
        self.classes[class_decl.name] = class_def
    
    def execute_block(self, statements: List[Statement], environment: Environment):
        """Execute a block of statements in the given environment"""
        previous = self.environment
        try:
            self.environment = environment
            
            for statement in statements:
                self.execute_statement(statement)
        finally:
            self.environment = previous
    
    def execute_statement(self, stmt: Statement):
        """Execute a single statement"""
        if isinstance(stmt, ExpressionStatement):
            self.evaluate(stmt.expression)
        
        elif isinstance(stmt, VariableDeclaration):
            value = None
            if stmt.initializer:
                value = self.evaluate(stmt.initializer)
            self.environment.define(stmt.name, value)
        
        elif isinstance(stmt, AssignmentStatement):
            value = self.evaluate(stmt.value)
            
            if stmt.operator == "=":
                self.environment.assign(stmt.name, value)
            elif stmt.operator == "&=":
                current = self.environment.get(stmt.name)
                if isinstance(current, str) and isinstance(value, str):
                    self.environment.assign(stmt.name, current + value)
                else:
                    raise RuntimeError(f"Cannot concatenate {type(current).__name__} and {type(value).__name__}")
        
        elif isinstance(stmt, MemberAssignmentStatement):
            obj = self.evaluate(stmt.object)
            value = self.evaluate(stmt.value)
            
            if isinstance(obj, ClassInstance):
                if stmt.operator == "=":
                    obj.set(stmt.property, value)
                elif stmt.operator == "&=":
                    current = obj.get(stmt.property) if obj.has_field(stmt.property) else ""
                    if isinstance(current, str) and isinstance(value, str):
                        obj.set(stmt.property, current + value)
                    else:
                        raise RuntimeError(f"Cannot concatenate {type(current).__name__} and {type(value).__name__}")
            else:
                raise RuntimeError(f"Cannot assign field to non-object")
        
        elif isinstance(stmt, IfStatement):
            condition_value = self.evaluate(stmt.condition)
            if self.is_truthy(condition_value):
                self.execute_block(stmt.then_block, Environment(self.environment))
            else:
                # Check elseif clauses
                executed = False
                for elseif_clause in stmt.elseif_clauses:
                    if self.is_truthy(self.evaluate(elseif_clause.condition)):
                        self.execute_block(elseif_clause.block, Environment(self.environment))
                        executed = True
                        break
                
                # Execute else block if no elseif was executed
                if not executed and stmt.else_block:
                    self.execute_block(stmt.else_block, Environment(self.environment))
        
        elif isinstance(stmt, ForStatement):
            # Initialize loop variable
            start_value = self.evaluate(stmt.start)
            loop_env = Environment(self.environment)
            loop_env.define(stmt.variable, start_value)
            
            # Execute loop
            while True:
                # Check condition
                prev_env = self.environment
                self.environment = loop_env
                
                try:
                    condition_value = self.evaluate(stmt.condition)
                    if not self.is_truthy(condition_value):
                        break
                    
                    # Execute body
                    self.execute_block(stmt.body, Environment(loop_env))
                    
                    # Execute increment
                    if stmt.increment:
                        self.evaluate(stmt.increment)
                
                finally:
                    self.environment = prev_env
        
        elif isinstance(stmt, ReturnStatement):
            value = None
            if stmt.value:
                value = self.evaluate(stmt.value)
            raise ReturnValue(value)
    
    def evaluate(self, expr: Expression) -> Any:
        """Evaluate an expression and return its value"""
        if isinstance(expr, LiteralExpression):
            return expr.value
        
        elif isinstance(expr, IdentifierExpression):
            return self.environment.get(expr.name)
        
        elif isinstance(expr, BinaryExpression):
            left = self.evaluate(expr.left)
            
            # Handle assignment operators
            if expr.operator == "=":
                if isinstance(expr.left, IdentifierExpression):
                    right = self.evaluate(expr.right)
                    self.environment.assign(expr.left.name, right)
                    return right
                else:
                    raise RuntimeError("Invalid assignment target")
            
            elif expr.operator == "&=":
                if isinstance(expr.left, IdentifierExpression):
                    right = self.evaluate(expr.right)
                    if isinstance(left, str) and isinstance(right, str):
                        new_value = left + right
                        self.environment.assign(expr.left.name, new_value)
                        return new_value
                    else:
                        raise RuntimeError(f"Cannot concatenate {type(left).__name__} and {type(right).__name__}")
                else:
                    raise RuntimeError("Invalid assignment target")
            
            # Regular binary operators
            right = self.evaluate(expr.right)
            
            if expr.operator == "+":
                return left + right
            elif expr.operator == "-":
                return left - right
            elif expr.operator == "*":
                return left * right
            elif expr.operator == "/":
                if right == 0:
                    raise RuntimeError("Division by zero")
                return left / right
            elif expr.operator == "%":
                return left % right
            elif expr.operator == "==":
                return left == right
            elif expr.operator == "!=":
                return left != right
            elif expr.operator == "<":
                return left < right
            elif expr.operator == "<=":
                return left <= right
            elif expr.operator == ">":
                return left > right
            elif expr.operator == ">=":
                return left >= right
            else:
                raise RuntimeError(f"Unknown binary operator: {expr.operator}")
        
        elif isinstance(expr, UnaryExpression):
            operand = self.evaluate(expr.operand)
            
            if expr.operator == "-":
                return -operand
            else:
                raise RuntimeError(f"Unknown unary operator: {expr.operator}")
        
        elif isinstance(expr, CallExpression):
            # Check if it's a built-in function
            if expr.name == "print":
                args = [self.evaluate(arg) for arg in expr.arguments]
                print(*args)
                return None
            
            # Look for function in current scope
            function = self.environment.get(expr.name)
            if callable(function):
                args = [self.evaluate(arg) for arg in expr.arguments]
                return function(*args)
            elif isinstance(function, Function):
                args = [self.evaluate(arg) for arg in expr.arguments]
                return function.call(self, args)
            else:
                raise RuntimeError(f"'{expr.name}' is not callable")
        
        elif isinstance(expr, NewExpression):
            if expr.class_name not in self.classes:
                raise RuntimeError(f"Undefined class '{expr.class_name}'")
            
            class_def = self.classes[expr.class_name]
            instance = class_def.instantiate()
            
            # Call constructor if it exists
            constructor = class_def.get_method("new")
            if constructor:
                func = Function(constructor, self.environment, instance)
                func.call(self, [])
            
            return instance
        
        elif isinstance(expr, MemberExpression):
            obj = self.evaluate(expr.object)
            
            if isinstance(obj, ClassInstance):
                if hasattr(obj, 'get'):
                    method = obj.get(expr.property)
                    if isinstance(method, FunctionDeclaration):
                        return Function(method, self.environment, obj)
                    return method
                else:
                    return getattr(obj, expr.property)
            else:
                raise RuntimeError(f"Cannot access property '{expr.property}' on non-object")
        
        elif isinstance(expr, MethodCallExpression):
            obj = self.evaluate(expr.object)
            
            if isinstance(obj, ClassInstance):
                method = obj.class_def.get_method(expr.method_name)
                if method:
                    func = Function(method, self.environment, obj)
                    args = [self.evaluate(arg) for arg in expr.arguments]
                    return func.call(self, args)
                else:
                    raise RuntimeError(f"Method '{expr.method_name}' not found")
            else:
                raise RuntimeError(f"Cannot call method on non-object")
        
        elif isinstance(expr, SuperCallExpression):
            # Super call: ..() - call parent method with same name
            try:
                current_method = self.environment.get("__current_method__")
                current_instance = self.environment.get("__current_instance__")
                
                if not current_method or not current_instance:
                    raise RuntimeError("Super call can only be used within a method")
                
                parent_method = current_instance.class_def.get_parent_method(current_method)
                if parent_method:
                    func = Function(parent_method, self.environment, current_instance)
                    args = [self.evaluate(arg) for arg in expr.arguments]
                    return func.call(self, args)
                else:
                    raise RuntimeError(f"No parent method '{current_method}' found for super call")
            except RuntimeError as e:
                if "Undefined variable" in str(e):
                    raise RuntimeError("Super call can only be used within a method")
                raise
        
        else:
            raise RuntimeError(f"Unknown expression type: {type(expr).__name__}")
    
    def is_truthy(self, value: Any) -> bool:
        """Determine if a value is truthy"""
        if value is None or value is False:
            return False
        if isinstance(value, (int, float)) and value == 0:
            return False
        if isinstance(value, str) and value == "":
            return False
        return True