import os
import uuid
import json
import boto3
import pickle
import shutil
import requests
import numpy as np
import librosalite
from io import BytesIO

if __name__ == '__main__':
    import tensorflow as tf
else:
    import tflite_runtime.interpreter as tflite


def most_similar(mp3tovecs,
                 weights,
                 positive=[],
                 negative=[],
                 noise=0,
                 vecs=None):
    mp3_vecs_i = np.array([weights[j] *
        np.sum([mp3tovecs[i, j] for i in positive] +
               [-mp3tovecs[i, j] for i in negative],
               axis=0) for j in range(len(weights))])
    if vecs is not None:
        mp3_vecs_i += np.sum(vecs, axis=0)
    if noise != 0:
        for mp3_vec_i in mp3_vecs_i:
            mp3_vec_i += np.random.normal(0, noise * np.linalg.norm(mp3_vec_i), mp3tovecs.shape[2])
    result = list(np.argsort(np.tensordot(mp3tovecs, mp3_vecs_i, axes=((1, 2), (0, 1)))))
    for i in negative:
        del result[result.index(i)]
    result.reverse()
    for i in positive:
        del result[result.index(i)]
    return result


def most_similar_by_vec(mp3tovecs,
                        weights,
                        positives=[],
                        negatives=[],
                        noise=0):
    mp3_vecs_i = np.array([weights[j] *
        np.sum(positives[j] if positives else [] +
               -negatives[j] if negatives else [],
               axis=0) for j in range(len(weights))])
    if noise != 0:
        for mp3_vec_i in mp3_vecs_i:
            mp3_vec_i += np.random.normal(0, noise * np.linalg.norm(mp3_vec_i), mp3tovecs.shape[2])
    result = list(np.argsort(-np.tensordot(mp3tovecs, mp3_vecs_i, axes=((1, 2), (0, 1)))))
    return result


# create a musical journey between given track "waypoints"
def join_the_dots(mp3tovecs, weights, ids, \
                  tracks, track_ids, track_indices, n=5, noise=0, replace=True):
    playlist = []
    playlist_tracks = [tracks[_] for _ in ids]
    end = start = ids[0]
    start_vec = mp3tovecs[track_indices[start]]
    for end in ids[1:]:
        end_vec = mp3tovecs[track_indices[end]]
        playlist.append(start)
        for i in range(n):
            candidates = most_similar_by_vec(mp3tovecs,
                                             weights,
                                             [[(n - i + 1) / n * start_vec[k] +
                                               (i + 1) / n * end_vec[k]]
                                              for k in range(len(weights))],
                                             noise=noise)
            for candidate in candidates:
                track_id = track_ids[candidate]
                if track_id not in playlist + ids and tracks[
                        track_id] not in playlist_tracks and tracks[
                            track_id][:tracks[track_id].find(' - ')] != tracks[
                                playlist[-1]][:tracks[playlist[-1]].find(' - ')]:
                    break
            playlist.append(track_id)
            playlist_tracks.append(tracks[track_id])
        start = end
        start_vec = end_vec
    playlist.append(end)
    return playlist_indices, playlist_tracks


def make_playlist(mp3tovecs, weights, playlist, \
                  tracks, track_ids, track_indices, size=10, lookback=3, noise=0):
    playlist_tracks = [tracks[_] for _ in playlist]
    playlist_indices = [track_indices[_] for _ in playlist]
    for i in range(len(playlist), size):
        candidates = most_similar(mp3tovecs,
                                  weights,
                                  positive=playlist_indices[-lookback:],
                                  noise=noise)
        for candidate in candidates:
            track_id = track_ids[candidate]
            if track_id not in playlist and tracks[
                    track_id] not in playlist_tracks and tracks[
                        track_id][:tracks[track_id].find(' - ')] != tracks[
                            playlist[-1]][:tracks[playlist[-1]].find(' - ')]:
                break
        playlist.append(track_id)
        playlist_tracks.append(tracks[track_id])
        playlist_indices.append(candidate)
    return playlist_indices, playlist_tracks


def get_similar_vec(track_url, interpreter):
    playlist_id = str(uuid.uuid4())
    sr = 22050
    n_fft = 2048
    hop_length = 512
    n_mels = interpreter.get_input_details()[0]['shape'][1]
    slice_size = interpreter.get_input_details()[0]['shape'][2]

    r = requests.get(track_url, allow_redirects=True)
    if r.status_code != 200:
        return []
    with open(f'/tmp/{playlist_id}.wav',
              'wb') as file:  # this is really annoying!
        shutil.copyfileobj(BytesIO(r.content), file, length=131072)
    y, sr = librosalite.load(f'/tmp/{playlist_id}.wav', mono=True)
    # cannot safely process two calls from same client
    os.remove(f'/tmp/{playlist_id}.wav')
    S = librosalite.melspectrogram(y=y,
                                   sr=sr,
                                   n_fft=n_fft,
                                   hop_length=hop_length,
                                   n_mels=n_mels,
                                   fmax=sr / 2)
    # hack because Spotify samples are a shade under 30s
    x = np.ndarray(shape=(1, n_mels, slice_size, 1),
                   dtype=np.float32)
    vecs = np.zeros(shape=(1, 100), dtype=np.float32)
    for slice in range(S.shape[1] // slice_size):
        log_S = librosalite.power_to_db(S[:, slice * slice_size:(slice + 1) *
                                          slice_size],
                                        ref=np.max)
        if np.max(log_S) - np.min(log_S) != 0:
            log_S = (log_S - np.min(log_S)) / (np.max(log_S) - np.min(log_S))
        x[0, :, :, 0] = log_S
        interpreter.set_tensor(interpreter.get_input_details()[0]['index'], x)
        interpreter.invoke()
        vecs += interpreter.get_tensor(interpreter.get_output_details()[0]['index'])
    # hack because Spotify samples are a shade under 30s
    log_S = librosalite.power_to_db(S[:, -slice_size:], ref=np.max)
    if np.max(log_S) - np.min(log_S) != 0:
        log_S = (log_S - np.min(log_S)) / (np.max(log_S) - np.min(log_S))
    x[-1, :, :, 0] = log_S
    interpreter.set_tensor(interpreter.get_input_details()[0]['index'], x)
    interpreter.invoke()
    vecs += interpreter.get_tensor(interpreter.get_output_details()[0]['index'])
    return vecs


def lambda_handler(event, context):
    """Sample pure Lambda function

    Parameters
    ----------
    event: dict, required
        API Gateway Lambda Proxy Input Format

        Event doc: https://docs.aws.amazon.com/apigateway/latest/developerguide/set-up-lambda-proxy-integrations.html#api-gateway-simple-proxy-for-lambda-input-format

    context: object, required
        Lambda Context runtime methods and attributes

        Context doc: https://docs.aws.amazon.com/lambda/latest/dg/python-context-object.html

    Returns
    ------
    API Gateway Lambda Proxy Output Format: dict

        Return doc: https://docs.aws.amazon.com/apigateway/latest/developerguide/set-up-lambda-proxy-integrations.html
    """

    # try:
    #     ip = requests.get("http://checkip.amazonaws.com/")
    # except requests.RequestException as e:
    #     # Send some context about this error to Lambda Logs
    #     print(e)

    #     raise e

    input_tracks = event.get('input_tracks', [])
    size = event.get('size', 10)
    creativity = event.get('creativity', 0.5)
    noise = event.get('noise', 0)
    lookback = event.get('lookback', 3)
    track_url = event.get('track_url', 'https://deej-ai.online/test.wav')

    s3 = boto3.resource('s3')
    mp3tovecs, track_indices, track_ids, tracks = pickle.loads(s3.Bucket('deej-ai.online').Object('stuff.p').get()['Body'].read())
    interpreter = tflite.Interpreter(model_content=s3.Bucket('deej-ai.online').Object('speccy_model.tflite').get()['Body'].read())
    interpreter.allocate_tensors()
    vec = get_similar_vec(track_url, interpreter)
    input_tracks = [track_ids[most_similar_by_vec(mp3tovecs[:, np.newaxis, 0, :], [1], [vec])[0]]]
    playlist_indices, playlist_tracks = make_playlist(mp3tovecs,
                                                      [creativity, 1 - creativity],
                                                      input_tracks,
                                                      tracks,
                                                      track_ids,
                                                      track_indices,
                                                      size=size,
                                                      lookback=lookback,
                                                      noise=noise)
    response = [[track_ids[_] for _ in playlist_indices], playlist_tracks]
    
    return {
        "statusCode": 200,
        "body": json.dumps(response),
    }


if __name__== '__main__':
    # tflite build from https://izhangzhihao.github.io/2020/03/17/Build-tflite-runtime-with-amazon-linux-1/ 
    model = tf.keras.models.load_model('../../deej-ai.online/speccy_model', custom_objects={'cosine_proximity' : tf.compat.v1.keras.losses.cosine_proximity})
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open('model.tflite', 'wb') as f:
        f.write(tflite_model)
        
    mp3tovecs = pickle.load(open('../../deej-ai.online/spotifytovec.p', 'rb'))
    mp3tovecs = dict(
        zip(mp3tovecs.keys(),
            [mp3tovecs[_] / np.linalg.norm(mp3tovecs[_]) for _ in mp3tovecs]))
    tracktovecs = pickle.load(open('../../deej-ai.online/tracktovec.p', 'rb'))
    tracktovecs = dict(
        zip(tracktovecs.keys(), [
            tracktovecs[_] / np.linalg.norm(tracktovecs[_])
            for _ in tracktovecs
        ]))
    track_indices = dict(map(lambda x: (x[1], x[0]), enumerate(mp3tovecs)))
    track_ids = [_ for _ in mp3tovecs]
    mp3tovecs = np.array([[mp3tovecs[_], tracktovecs[_]] for _ in mp3tovecs])
    tracks = pickle.load(open('../../deej-ai.online/spotify_tracks.p', 'rb'))
    stuff = (mp3tovecs, track_indices, track_ids, tracks)
    pickle.dump(stuff, open('stuff.p', 'wb'))

    s3 = boto3.resource('s3')
    s3.Bucket('deej-ai.online').upload_file('stuff.p', 'stuff.p')
    s3.Bucket('deej-ai.online').upload_file('model.tflite', 'speccy_model.tflite')