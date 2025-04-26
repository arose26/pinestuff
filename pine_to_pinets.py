class Node:
    def __repr__(self):
        attrs = {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
        return f"{self.__class__.__name__}({', '.join(f'{k}={v!r}' for k, v in attrs.items())})"

class Script(Node):
    def __init__(self, body, annotations):
        self.body = body
        self.annotations = annotations

class ReAssign(Node):
    def __init__(self, target, value):
        self.target = target
        self.value = value

class Assign(Node):
    def __init__(self, target, value, annotations):
        self.target = target
        self.value = value
        self.annotations = annotations

class Name(Node):
    def __init__(self, id, ctx):
        self.id = id
        self.ctx = ctx # Store() or Load() -> useful for Assign vs. Use

class Constant(Node):
    def __init__(self, value):
        self.value = value

class BinOp(Node):
    def __init__(self, left, op, right):
        self.left = left
        self.op = op # Add(), Sub() etc.
        self.right = right

class Add: pass
class Sub: pass
class Mult: pass
class Div: pass
class Mod: pass
class Gt: pass
class Lt: pass
class Eq: pass
class GtE: pass
class Or: pass
class And: pass
class Store: pass
class Load: pass

class FunctionDef(Node):
    def __init__(self, name, args, body, method=None, export=None, annotations=[]):
        self.name = name
        self.args = args
        self.body = body
        self.method = method
        self.export = export
        self.annotations = annotations

class While(Node):
    def __init__(self, test, body=None):
        self.test = test
        self.body = body

class If(Node):
    def __init__(self, test, body, orelse):
        self.test = test
        self.body = body
        self.orelse = orelse

class ForTo(Node):
    def __init__(self, target, start, end, body):
        self.target = target
        self.start = start
        self.end = end
        self.body = body

class Param(Node):
    def __init__(self, name):
        self.name = name

class UnaryOp(Node):
    def __init__(self, op, operand):
        self.op = op # UAdd(), USub() etc.
        self.operand = operand

class Call(Node):
    def __init__(self, func, args):
        self.func = func
        self.args = args

class Attribute(Node):
    def __init__(self, value, attr, ctx):
        self.value = value
        self.attr = attr
        self.ctx = ctx

class Arg(Node):
    def __init__(self, value, name=None):
        self.value = value
        self.name = name

class Expr(Node):
    def __init__(self, value):
        self.value = value

class Conditional(Node):
    def __init__(self, test, body, orelse):
        self.test = test
        self.body = body
        self.orelse = orelse

class BoolOp(Node):
    def __init__(self, op, values):
        self.op = op
        self.values = values

class Compare(Node):
    def __init__(self, left, ops, comparators):
        assert len(ops) == 1 and len(comparators) == 1
        self.left = left
        self.op = ops[0]
        self.right = comparators[0]

class Subscript(Node):
    def __init__(self, value, slice, ctx):
        self.value = value
        self.slice = slice
        self.ctx = ctx

# --- ESTree (JavaScript AST) Node Representations (Simplified) ---

def estree_node(type, **kwargs):
    node = {'type': type}
    node.update(kwargs)
    return node

# --- Transpiler Logic ---

class PyneToJsAstConverter:
    def __init__(self):
        self._declared_vars = set() # Track declared variables

    def _map_operator(self, op_node):
        if isinstance(op_node, Add): return '+'
        elif isinstance(op_node, Sub): return '-'
        elif isinstance(op_node, Mult): return '*'
        elif isinstance(op_node, Div): return '/'
        elif isinstance(op_node, Mod): return '%'
        # Add mappings for other operators as needed
        raise NotImplementedError(f"Operator mapping not implemented for {type(op_node)}")

    def _map_comparison_operator(self, op_node):
        if isinstance(op_node, GtE): return '>='
        elif isinstance(op_node, Gt): return '>'
        elif isinstance(op_node, Lt): return '<'
        elif isinstance(op_node, Eq): return '==='
        # Add mappings for other comparison operators (NotEq, LtE) if needed
        raise NotImplementedError(f"Comparison operator mapping not implemented for {type(op_node)}")

    def _map_logical_operator(self, op_node):
        if isinstance(op_node, Or): return '||'
        elif isinstance(op_node, And): return '&&'
        raise NotImplementedError(f"Logical operator mapping not implemented for {type(op_node)}")

    def visit(self, node):
        method_name = 'visit_' + type(node).__name__
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node):
        raise NotImplementedError(f"No visit method implemented for {type(node)}")

    def visit_Script(self, node):
        body = [self.visit(stmt) for stmt in node.body]
        # Filter out None results if some statements don't produce output
        body = [stmt for stmt in body if stmt]
        return estree_node('Program', body=body, sourceType='module') # Or 'script'

    def visit_Assign(self, node):
        var_name = node.target.id
        js_value = self.visit(node.value)

        if var_name not in self._declared_vars:
            self._declared_vars.add(var_name)
            # First assignment -> VariableDeclaration
            declaration = estree_node('VariableDeclarator',
                                      id=self.visit(node.target),
                                      init=js_value)
            # Using 'const' now. Note: This assumes variables assigned with '='
            # in PineScript are not later reassigned with ':='.
            # A more robust implementation might require multi-pass analysis.
            return estree_node('VariableDeclaration', declarations=[declaration], kind='const')
        else:
            # Subsequent assignment -> AssignmentExpression
            # This case might be less common if PineScript prefers ':=' for reassignment
            return estree_node('ExpressionStatement',
                               expression=estree_node('AssignmentExpression',
                                                    operator='=',
                                                    left=self.visit(node.target),
                                                    right=js_value))

    def visit_ReAssign(self, node):
        # Treat ReAssign like Assign, but assume var is already declared
        var_name = node.target.id
        js_value = self.visit(node.value) # Evaluate the right-hand side

        if var_name not in self._declared_vars:
           # This case might indicate an issue if ReAssign is used before Assign
           # For now, let's declare it, but maybe raise warning/error later
           print(f"Warning: ReAssign used for potentially undeclared variable '{var_name}'. Declaring with 'let'.")
           self._declared_vars.add(var_name)
           declaration = estree_node('VariableDeclarator',
                                     id=self.visit(node.target),
                                     init=js_value)
           # Use 'let' here because ReAssign implies potential future reassignments
           return estree_node('VariableDeclaration', declarations=[declaration], kind='let')
        else:
             # Variable already declared, generate an assignment expression
             return estree_node('ExpressionStatement',
                               expression=estree_node('AssignmentExpression',
                                                    operator='=',
                                                    left=self.visit(node.target),
                                                    right=js_value))

    def visit_Name(self, node):
        # Both loading and storing map to an Identifier in JS AST
        return estree_node('Identifier', name=node.id)

    def visit_Constant(self, node):
        # Handle different types of constants if needed (strings, bools, etc.)
        return estree_node('Literal', value=node.value, raw=repr(node.value))

    def visit_BinOp(self, node):
        return estree_node('BinaryExpression',
                           operator=self._map_operator(node.op),
                           left=self.visit(node.left),
                           right=self.visit(node.right))

    def visit_Call(self, node):
        callee = self.visit(node.func)

        # Check if this is a call to the 'input' function identifier
        is_input_call = isinstance(node.func, Name) and node.func.id == 'input'

        positional_args_js = []
        named_args_props = []

        for i, arg in enumerate(node.args):
            arg_value_js = self.visit(arg.value)
            if arg.name:
                # Named argument: Create an ESTree Property
                prop = estree_node('Property',
                                   key=estree_node('Identifier', name=arg.name),
                                   value=arg_value_js,
                                   kind='init',
                                   method=False,
                                   shorthand=False,
                                   computed=False)
                named_args_props.append(prop)
            else:
                # Positional argument
                positional_args_js.append(arg_value_js)

                # --- Specialization logic for 'input' ---
                if is_input_call and i == 0 and isinstance(arg.value, Constant):
                    # Check the type of the *value* of the first positional Constant argument
                    first_arg_py_value = arg.value.value
                    input_type_attr = None

                    # Order matters: check float before int if Python auto-converts int to float visually (e.g. 2.0)
                    # However, Python AST Constant should preserve the original type
                    if isinstance(first_arg_py_value, bool):
                        input_type_attr = 'bool'
                    elif isinstance(first_arg_py_value, float):
                        input_type_attr = 'float'
                    elif isinstance(first_arg_py_value, int): # Check int after float
                        input_type_attr = 'int'
                    # Add other types like string if needed:
                    # elif isinstance(first_arg_py_value, str):
                    #     input_type_attr = 'string'

                    if input_type_attr:
                        # Modify the callee from Identifier('input')
                        # to MemberExpression('input', 'type')
                        callee = estree_node('MemberExpression',
                                             object=estree_node('Identifier', name='input'), # Recreate base 'input' identifier
                                             property=estree_node('Identifier', name=input_type_attr),
                                             computed=False)
                # --- End specialization logic ---


        final_args_js = positional_args_js

        # If there were named arguments, create an ObjectExpression and add it
        if named_args_props:
            options_object = estree_node('ObjectExpression', properties=named_args_props)
            final_args_js.append(options_object)

        # Use the potentially modified callee
        return estree_node('CallExpression', callee=callee, arguments=final_args_js)

    def visit_Attribute(self, node):
        # Maps PineScript's `object.attribute` to JS `object.property`
        return estree_node('MemberExpression',
                           object=self.visit(node.value),
                           property=estree_node('Identifier', name=node.attr),
                           computed=False) # Assuming dot notation access

    def visit_Expr(self, node):
        # An Expr node in Python AST usually represents an expression used as a statement
        # (e.g., a function call that doesn't assign to anything).
        # In ESTree, this corresponds to an ExpressionStatement.

        # Special cases for control flow nodes - return them directly without wrapping
        if isinstance(node.value, (While, If, ForTo)):
            return self.visit(node.value)

        return estree_node('ExpressionStatement', expression=self.visit(node.value))

    def visit_Conditional(self, node):
        # Maps to JavaScript's ternary operator
        return estree_node('ConditionalExpression',
                           test=self.visit(node.test),
                           consequent=self.visit(node.body),
                           alternate=self.visit(node.orelse))

    def visit_BoolOp(self, node):
        # Maps to JavaScript's LogicalExpression (||, &&)
        # Assumes left-associativity for operators like || and &&
        if len(node.values) < 2:
             raise ValueError("BoolOp requires at least two values")

        # Build nested logical expressions: a || b || c -> ((a || b) || c)
        expression = estree_node('LogicalExpression',
                                 operator=self._map_logical_operator(node.op),
                                 left=self.visit(node.values[0]),
                                 right=self.visit(node.values[1]))

        for i in range(2, len(node.values)):
            expression = estree_node('LogicalExpression',
                                     operator=self._map_logical_operator(node.op),
                                     left=expression,
                                     right=self.visit(node.values[i]))
        return expression

    def visit_Compare(self, node):
         # Maps to JavaScript's BinaryExpression with comparison operators
         return estree_node('BinaryExpression',
                            operator=self._map_comparison_operator(node.op),
                            left=self.visit(node.left),
                            right=self.visit(node.right))

    def visit_Subscript(self, node):
        # Maps PineScript's series access (e.g., close[1]) to JS MemberExpression
        # Assuming the slice is usually a Constant integer for historical access
        obj = self.visit(node.value)
        prop = self.visit(node.slice) # Visit the slice (e.g., Constant(1))

        # In ESTree, array access like obj[1] is a MemberExpression
        # with computed=True and the property being the index expression.
        return estree_node('MemberExpression',
                           object=obj,
                           property=prop,
                           computed=True) # Use bracket notation

    def visit_FunctionDef(self, node):
        # Convert function name to identifier
        func_name = node.name

        # Convert parameters
        params = [self.visit(arg) for arg in node.args]

        # Convert body statements
        body_statements = [self.visit(stmt) for stmt in node.body]

        # Create function body block
        body_block = estree_node('BlockStatement', body=body_statements)

        # Check if the last statement is an expression that could be returned
        if body_statements and isinstance(node.body[-1], Expr):
            # Replace the last expression statement with a return statement
            return_stmt = estree_node('ReturnStatement',
                                    argument=self.visit(node.body[-1].value))
            body_block['body'] = body_statements[:-1] + [return_stmt]

        # Create the function declaration
        func_declaration = estree_node(
            'VariableDeclaration',
            declarations=[
                estree_node(
                    'VariableDeclarator',
                    id=estree_node('Identifier', name=func_name),
                    init=estree_node(
                        'ArrowFunctionExpression',
                        id=None,
                        params=params,
                        body=body_block,
                        expression=False,
                        generator=False,
                        **{"async":False}
                    )
                )
            ],
            kind='const'
        )

        # Add function name to declared variables
        self._declared_vars.add(func_name)

        return func_declaration

    def visit_Param(self, node):
        # Convert parameter name to an Identifier node
        # Parameters in JavaScript functions are represented as simple identifiers
        return estree_node('Identifier', name=node.name)

    def visit_While(self, node):
        # Convert the test condition
        test_js = self.visit(node.test)

        # Convert the body statements
        body_statements = [self.visit(stmt) for stmt in node.body]
        # Filter out None results if some statements don't produce output
        body_statements = [stmt for stmt in body_statements if stmt]

        # Create a BlockStatement for the body
        body_block = estree_node('BlockStatement', body=body_statements)

        # Create and return the WhileStatement
        # Using a dictionary with type field instead of named parameters
        return {
            'type': 'WhileStatement',
            'test': test_js,
            'body': body_block
        }

    def visit_ForTo(self, node):
        # Get the loop variable name and create an identifier
        var_name = node.target.id
        var_id = self.visit(node.target)

        # Convert start and end expressions
        start_js = self.visit(node.start)
        end_js = self.visit(node.end)

        # Create initialization: let i = start
        init = estree_node('VariableDeclaration',
                          declarations=[
                              estree_node('VariableDeclarator',
                                         id=var_id,
                                         init=start_js)
                          ],
                          kind='let')

        # Add variable to declared variables
        self._declared_vars.add(var_name)

        # Create test condition: i <= end
        test = estree_node('BinaryExpression',
                          operator='<=',
                          left=var_id,
                          right=end_js)

        # Create update expression: i++
        update = estree_node('UpdateExpression',
                            operator='++',
                            argument=var_id,
                            prefix=False)

        # Convert the body statements
        body_statements = [self.visit(stmt) for stmt in node.body]
        # Filter out None results if some statements don't produce output
        body_statements = [stmt for stmt in body_statements if stmt]

        # Create a BlockStatement for the body
        body_block = estree_node('BlockStatement', body=body_statements)

        # Create and return the ForStatement
        # Using a dictionary with type field instead of named parameters
        return {
            'type': 'ForStatement',
            'init': init,
            'test': test,
            'update': update,
            'body': body_block
        }

    def visit_If(self, node):
        # Convert the test condition
        test_js = self.visit(node.test)

        # Convert the body statements
        body_statements = [self.visit(stmt) for stmt in node.body]
        # Filter out None results if some statements don't produce output
        body_statements = [stmt for stmt in body_statements if stmt]

        # Create a BlockStatement for the body
        consequent_block = estree_node('BlockStatement', body=body_statements)

        # Handle the else branch if it exists
        alternate_block = None
        if node.orelse:
            # Check if it's a list (else branch) or another If node (else if branch)
            if isinstance(node.orelse, list):
                # Convert the else branch statements
                else_statements = [self.visit(stmt) for stmt in node.orelse]
                # Filter out None results if some statements don't produce output
                else_statements = [stmt for stmt in else_statements if stmt]

                # Create a BlockStatement for the else branch
                alternate_block = estree_node('BlockStatement', body=else_statements)
            elif isinstance(node.orelse, If):
                # It's an else if branch, recursively visit it
                alternate_block = self.visit(node.orelse)
            else:
                # Unexpected type
                raise ValueError(f"Unexpected type for else branch: {type(node.orelse)}")

        # Create and return the IfStatement
        # Using a dictionary with type field instead of named parameters
        return {
            'type': 'IfStatement',
            'test': test_js,
            'consequent': consequent_block,
            'alternate': alternate_block
        }


""" aka PineStuff/pine_to_pinets.py """
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python transpilev1.py <filename>")
        sys.exit(1)
    filename = sys.argv[1]

    print('Dumping PineScript -> AST...')
    from pynescript.ast import dump
    from pynescript.ast import parse
    with open(filename, "r") as f:
        tree = parse(f.read())
    tree_dump = dump(tree, indent=2)
    print('Dump completed. Converting AST -> JS...')


    # Create converter instance and transpile Python AST to JS AST
    converter = PyneToJsAstConverter()
    js_ast = converter.visit(eval(tree_dump))

    # Transpile the JS AST to JavaScript
    import escodegen
    formatted_code = escodegen.generate(js_ast)

    if len(sys.argv) > 2:
        filename = sys.argv[2]
        with open(filename, "w") as f:
            f.write(formatted_code)
    else:
        print(formatted_code)

