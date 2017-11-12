import tensorflow as tf
import functools
    
# A Component builds a graph with variables.

# A Model is a collection of Components with a training procedure.

# A component is a graph that contains variables that will be reused.
def component(build_component):
    
    class Component:
        def __init__(self, **auto_kwargs):
            self.auto_kwargs = auto_kwargs
        
        def __call__(self, *args, **kwargs):
            if self.auto_kwargs:
                auto_kwargs = self.auto_kwargs.copy()
                auto_kwargs.update(kwargs)
                kwargs = auto_kwargs
            return build_component(*args, **kwargs)
    
    Component.__name__ = build_component.__name__
    return model(Component, default_method='__call__')

# Wraps a function so that variables are namespaced.
def op(f):
    @functools.wraps(f)
    def f_with_scope(*args, **kwargs):
        with tf.variable_scope(None, default_name=f.__name__) as scope:
            with tf.name_scope(scope.name + '/'):
                return f(*args, **kwargs)
    return f_with_scope

# Wraps a class method so that variables are namespaced.
def model(cls, default_method='__init__'):
    
    # Decorate every class method.
    for name in cls.__dict__:
        method = getattr(cls, name)
        if callable(method):
            if name == default_method:
                setattr(cls, name, scope_method(method, None))
            else:
                setattr(cls, name, scope_method(method, name))

    @property
    def vars(self):
        return get_variables(self._scope.name + '/')
    cls.vars = vars
    
    return cls

def scope_method(method, name):

    @functools.wraps(method)
    def wrapped_method(self, *args, **kwargs):
        # Initialize the class scope
        if not hasattr(self, '_scope'):
            with tf.variable_scope(None, default_name=self.__class__.__name__) as scope:
                self._scope = scope
                self._subscopes = {}
        
        # Initialize the method scope
        if method not in self._subscopes:
            with tf.variable_scope(self._scope):
                with tf.name_scope(self._scope.name + '/'):
                    if name:
                        with tf.variable_scope(None, default_name=name) as method_scope:
                            with tf.name_scope(method_scope.name + '/'):
                                out = method(self, *args, **kwargs)
                    else:
                        method_scope = self._scope
                        out = method(self, *args, **kwargs)
            self._subscopes[method] = method_scope
            return out

        method_scope = self._subscopes[method]
        with tf.variable_scope(method_scope, reuse=True):
            with tf.name_scope(method_scope.name + '/'):
                return method(self, *args, **kwargs)

    return wrapped_method

def get_variables(scope=None):
    return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
