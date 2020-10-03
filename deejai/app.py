# TODO
# name audio bucket or get created name
# limit audio to be between 5 and 30 seconds
# add spotify token to object name
# create spotify playlist
# gui

import os
import uuid
import json
import boto3
import urllib
import pickle
import shutil
import spotipy
import requests
import numpy as np
import librosalite
from io import BytesIO
import spotipy.util as util

if __name__ == '__main__':
    import tensorflow as tf
else:
    import tflite_runtime.interpreter as tflite

size = 20
creativity = 0.5
noise = 0
lookback = 3


def add_tracks_to_playlist(sp, username, playlist_id, track_ids, replace=False):
    if sp is not None and username is not None and playlist_id is not None:
        try:
            if replace:
                sp.user_playlist_replace_tracks(username, playlist_id, track_ids)
            else:
                sp.user_playlist_add_tracks(username, playlist_id, track_ids)
        except spotipy.client.SpotifyException:
            pass


def user_playlist_create(sp,
                         username,
                         playlist_name,
                         description='',
                         public=True):
    data = {
        'name': playlist_name,
        'public': public,
        'description': description
    }
    return sp._post("users/%s/playlists" % (username, ), payload=data)['id']


def make_spotify_playlist(token,
                          username,
                          tracks,
                          playlist_name='Deej-A.I.'):
    playlist_id = None
    if playlist_name != '':
        sp = spotipy.Spotify(token)
        if sp is not None:
            try:
                playlists = sp.user_playlists(username)
                if playlists is not None:
                    playlist_ids = [
                        playlist['id'] for playlist in playlists['items']
                        if playlist['name'] == playlist_name
                    ]
                    if len(playlist_ids) > 0:
                        playlist_id = playlist_ids[0]
                    else:
                        if 'tracks' in content:
                            # spotipy create_user_playlist is broken
                            playlist_id = user_playlist_create(
                                sp, username, playlist_name,
                                'Created by Deej-A.I. http://deej-ai.online'
                            )
            except:
                pass
        if playlist_id is None:
            print(f'Unable to access playlist {playlist_name} for user {username}')
            return ''
        else:
            print(f'Playlist {playlist_id}')
            add_tracks_to_playlist(sp, username, playlist_id, tracks, replace=True)
            return playlist_id


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


def get_similar_vec(s3, bucket, key, interpreter):
    playlist_id = str(uuid.uuid4())
    sr = 22050
    n_fft = 2048
    hop_length = 512
    n_mels = interpreter.get_input_details()[0]['shape'][1]
    slice_size = interpreter.get_input_details()[0]['shape'][2]

    s3.Bucket(bucket).download_file(key, f'/tmp/{playlist_id}.wav')
    y, sr = librosalite.load(f'/tmp/{playlist_id}.wav', mono=True)
    os.remove(f'/tmp/{playlist_id}.wav')
    S = librosalite.melspectrogram(y=y,
                                   sr=sr,
                                   n_fft=n_fft,
                                   hop_length=hop_length,
                                   n_mels=n_mels,
                                   fmax=sr / 2)
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
    log_S = librosalite.power_to_db(S[:, -slice_size:], ref=np.max)
    if np.max(log_S) - np.min(log_S) != 0:
        log_S = (log_S - np.min(log_S)) / (np.max(log_S) - np.min(log_S))
    x[-1, :, :, 0] = log_S
    interpreter.set_tensor(interpreter.get_input_details()[0]['index'], x)
    interpreter.invoke()
    vecs += interpreter.get_tensor(interpreter.get_output_details()[0]['index'])
    return vecs


def lambda_handler(event, context):
    s3 = boto3.resource('s3')
    bucket = event['Records'][0]['s3']['bucket']['name']
    key = urllib.parse.unquote_plus(event['Records'][0]['s3']['object']['key'])
    username_and_token = key[key.find('_') + 1: -4]
    token = username_and_token[username_and_token.find('_') + 1:]
    username = username_and_token[: username_and_token.find('_')]
    print(username, token)
    playlist_id = ''
    
    try:
        mp3tovecs, track_indices, track_ids, tracks = pickle.loads(s3.Bucket('deej-ai.online').Object('stuff.p').get()['Body'].read())
        interpreter = tflite.Interpreter(model_content=s3.Bucket('deej-ai.online').Object('speccy_model.tflite').get()['Body'].read())
        interpreter.allocate_tensors()
        vec = get_similar_vec(s3, bucket, key, interpreter)
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
        print(response)
        playlist_id = make_spotify_playlist(token, username, response[0])
        s3.Object('deej-ai.online', key).put(Body=playlist_id)
              
    except Exception as e:
        print(e)
        
    finally:
        s3.Object(bucket, key).delete()
    
    return {
        "statusCode": 200,
        "body": json.dumps(playlist_id),
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