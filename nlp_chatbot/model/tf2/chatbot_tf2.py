import json
import os
import numpy as np
import tensorflow as tf

from . import encoder
from . import model
from . import sample




def interact_model(
    temperature,
    top_k,
    top_p,
    nsamples,
    batch_size,
    length,
    input_utt,
    raw_text,
    seed=None,
):

    models_dir = os.path.join(os.getcwd(), 'nlp_chatbot/model/pretrained_models')
    if batch_size is None:
        batch_size = 1
    assert nsamples % batch_size == 0

    enc = encoder.get_encoder('345M_org', models_dir)
    hparams = model.default_hparams()
    with open(os.path.join(models_dir, '345M_org', 'hparams.json')) as f:
        hparams.update(json.load(f))

    with tf.compat.v1.Session(graph=tf.Graph()) as sess:
        contxt = tf.compat.v1.placeholder(tf.int32, [batch_size, None])
        np.random.seed(seed)
        tf.compat.v1.set_random_seed(seed)
        output = sample.sample_sequence(
            hparams=hparams, length=length,
            context=contxt,
            batch_size=batch_size,
            temperature=temperature, top_k=top_k, top_p=top_p
        )

        saver = tf.compat.v1.train.Saver()
        ckpt = tf.train.latest_checkpoint(os.path.join(models_dir, '345M_org'))
        saver.restore(sess, ckpt)

        
        
        raw_text +='\n' + 'user: '+ input_utt + '\n' + 'bot: '

        contxt_tokens = enc.encode(raw_text)
        for _ in range(nsamples // batch_size):
            out = sess.run(output, feed_dict={
                contxt: [contxt_tokens for _ in range(batch_size)]
            })[:, len(contxt_tokens):]
            for i in range(batch_size):
                text = enc.decode(out[i])
                result=list(text.partition('\n'))
                raw_text += str(result[0])
        return result[0], raw_text
