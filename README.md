# Image Similarity Using Siamese
<b>Siamese Networks</b> are neural networks which share weights between two or more sister networks, each producing embedding vectors of its respective inputs.

In supervised similarity learning, the network are then trained to maximize the contrast(distance) between embeddings of inputs of different classes, while minimizing the distance between embeddings of similar classes, resulting in embedding spaces that reflect the class segmentation of the training inputs.

<b>Training Images</b><br>
![pairs_image_label_1](https://user-images.githubusercontent.com/15634495/161435960-5b029bce-8a49-4f4f-aa91-72100e47d020.png) <br>

<b>Validation Images</b><br>
![pairs_image_label_1_val](https://user-images.githubusercontent.com/15634495/161436143-af8624ea-fb51-48dc-80f2-89f810b9b3a4.png) <br>

<b>Test Images</b><br>
![pairs_image_label_1_test](https://user-images.githubusercontent.com/15634495/161436291-ce80258c-d5d4-4f89-b8d6-bcd7c0201e74.png)<br>

<b>Accuracy</b></br>
![Accuracy](https://user-images.githubusercontent.com/15634495/161436339-18691e1d-c53d-4970-b27d-445ac64a90f2.png)<br>

<b>Loss</b><br>
![Loss](https://user-images.githubusercontent.com/15634495/161436369-56b17517-8166-4e78-8391-a81f76f030a3.png)<br>


<b>Contastive Loss</b><br>
```
def loss(margin=1):
  """Provides 'contrastive_loss' an encoding scope with variable 'margin'.
  Arguments:
    margin: Integer, defines the baseline for distance for which pairs
            should be classfied as dissimilar. - (default is 1)

  Returns:
    'contrastive loss' function with data ('margin') attached
  """
  # Contrastive loss = mean( (1-true_value) * square(prediction) +
  #                         true_value * square( max(margin-prediction, 0)))
  def constrastive_loss(y_true, y_pred):
    """Calculate the constrastive loss.
    Arguments:
      y_true: List of labels, each labels is of type float32.
      y_pred: List of predictions of same length as of y_true,
              each label is of type float 32

    Returns: 
      A tensors contaning constrastive loss as floating point value
    """
    square_pred = tf.math.square(y_pred)
    margin_square = tf.math.square(tf.math.maximum(margin - (y_pred), 0))
    return tf.math.reduce_mean(
        (1-y_true) * square_pred + (y_true) * margin_square
    )

  return constrastive_loss
  ```
  
  <b>Evaluation </b><br>
  ```
  625/625 [==============================] - 2s 3ms/step - loss: 0.0116 - accuracy: 0.9850
test loss, test acc: [0.011630297638475895, 0.9849500060081482]
```
