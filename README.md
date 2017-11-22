# deep

An attempt to make TensorFlow scopes and variable sharing more Pythonic by replacing them with two concepts: **ops** and **components**. An **op** is a template for building a part of a graph, whereas a **component** encapsulates both a group of graph fragments *and a particular instantiation of their variables*.

## Ops

An **op** is a function that builds a fragment of a graph; its arguments should correspond to the fragment's inputs. There is no concept of variable sharing at this point.

    from deep import op
  
    @op
    def layer(x, num_outputs):
      A = tf.get_variable('weights', shape=[None, x.shape[1], num_outputs])
      b = tf.get_variable('bias', shape=[None, 1])
      return tf.nn.relu(tf.matmul(x, A) + b)

    y = layer(layer(layer(x, 8), 8), 1)

In the above example, each call to `layer` creates a new set of variables. This is more or less how you'd expect a pure Python function to behave.

## Components

A **component** represents a collection of graph fragments, and an instance of the component corresponds to an instantiation of variables.

    from deep import component
  
    @component
    class BinaryClassifier:
      def classify(self, x):
        out = x
        out = layer(out, 8)
        out = layer(out, 8)
        out = layer(out, 8)
        out = layer(out, 1)
        return tf.nn.log_sigmoid(out)
      
      def loss(self, x, y):
        logits = self.classify(x)
        return tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=y)

    # train a classifier
    classifier = BinaryClassifier()
    loss = classifier.loss(x_train, y_train)
    ...  # minimize loss
    y_pred = classifier.classify(x_test)  # use the classifier on test data

Any variables created in `classify` are automatically reused whenever it is called. However, variables are only reused when dealing with the same class instance. Constructing a second classifier is no problem:

    # train a second classifier on a different dataset
    classifier2 = BinaryClassifier()
    loss2 = classifier2.loss(x2_train, y2_train)
    ...  # minimize loss2
    classifier2.classify(x2_test)  # use the classifier on test data

The two classifier instances do not interfere with each other.

As a second example, 

    from deep import component, variables
  
    @component
    class GAN:
      def generate(self, z):
        return tf.layers.conv2d(...)
      
      def discriminate(self, x):
        return tf.nn.sigmoid(...)
      
      def loss(self, x):
        x_fake = self.generate(tf.random_normal(...))
        return tf.log(self.discriminate(x)) + tf.log(1 - self.discriminate(x_fake))
        
    gan = GAN()
    loss = gan.loss(x)
    train_disc_op = optimizer.minimize(loss, var_list=variables(gan.generate))
    train_gen_op = optimizer.minimize(-loss, var_list=variables(gan.discriminate))
        
Any variables created in a particular method are reused when that method is called again. `variables` returns the list of trainable variables belonging to that method.
