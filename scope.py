import tensorflow as tf
import functools

def op(f):
    """Wraps a function so that variables are namespaced."""
    @functools.wraps(f)
    def f_with_scope(*args, **kwargs):
        with tf.variable_scope(None, default_name=f.__name__) as scope:
            with tf.name_scope(scope.name + '/'):
                return f(*args, **kwargs)
    return f_with_scope


def component(cls):
    """Wraps a class so that variables are namespaced."""
    default_method = None
    #if '__init__' in cls.__dict__:
    #    default_method = '__init__'
    #elif '__call__' in cls.__dict__:
    #    default_method = '__call__'
    # This is dangerous, since if you add an __init__ method, your old models stop working.
    
    # Decorate every class method.
    for name in cls.__dict__:
        method = getattr(cls, name)
        if callable(method):
            if name == default_method:
                setattr(cls, name, scope_method(method, None))
            else:
                setattr(cls, name, scope_method(method, name))
    
    # Copy from Component.
    for name in Component.__dict__:
        method = getattr(cls, name)
        if callable(method):
            setattr(cls, name, method)

    return cls

class Component:
    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self._scope.name + '/')

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

#def get_hyperparam(*args, **kwargs):
#    kwargs['trainable'] = False
#    if 'collections' not in kwargs:
#        kwargs['collections'] = [tf.GraphKeys.GLOBAL_VARIABLES]
#    kwargs['collections'].add('hyperparams')
#    return tf.get_variable(*args, **kwargs)
