# standard imports
import pandas as pd
import numpy as np
import os
import glob
import pickle
import matplotlib.pyplot as plt
# flask imports
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
# ML helpers
from collections import Counter
import librosa
import librosa.display as _display
import xgboost as xgb
from IPython.display import clear_output
from scipy.fftpack import fft, ifft
from itertools import compress

app = Flask(__name__)

@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r

def findsilence(y,sr,ind_i):
    hop = int(round(sr*0.18)) #hop and width defines search window
    width = int(sr*0.18)
    n_slice = int(len(y)/hop)
    starts = np.arange(n_slice)*hop
    ends = starts+width
    if hop != width:
        cutoff = np.argmax(ends>len(y))
        starts = starts[:cutoff]
        ends = ends[:cutoff]
        n_slice = len(starts)
    mask = [i for i in map(lambda i: np.dot(y[starts[i]:ends[i]],y[starts[i]:ends[i]])/width < (0.5 * np.dot(y,y)/len(y)), range(n_slice))]
    starts =  list(compress(starts+ind_i,mask))
    ends = list(compress(ends+ind_i,mask))
    return zip(starts,ends)
 
def merger(tulist):
    tu=[]
    for tt in tulist:
        tu.append(tt)
    tu = tuple(tu)
    cnt = Counter(tu)
    res = [i for i in filter(lambda x: cnt[x]<2, tu)]
#     return res
#     return [i for i in map(lambda x: tuple(x),np.array(res).reshape((19,2)))]
    return [i for i in map(lambda x: tuple(x),np.array(res).reshape(int(len(res)),2))]

def shade_silence(filename,start=0,end=None,disp=True,output=False, itr='', save=None):
    """Find signal (as output) or silence (as shaded reagion  in plot) in a audio file
    filename: (filepath) works best with .wav format
    start/end: (float or int) start/end time for duration of interest in second (default= entire length)
    disp: (bool) whether to display a plot(default= True)
    output: (bool) whether to return an output (default = False)
    itr: (int) iteration use for debugging purpose
    save: (str) filename to save to
    """
    try:
        y, sr = librosa.load(filename)
    except:
        pass
        # obj = thinkdsp.read_wave(filename)
        # y = obj.ys
        # sr = obj.framerate
        # print(itr, ' : librosa.load failed for '+filename)

    t = np.arange(len(y))/sr

    i = int(round(start * sr))
    if end != None:
        j = int(round(end * sr))
    else:
        j = len(y)
    fills = findsilence(y[i:j],sr,i)
    if disp:
        fig, ax = plt.subplots(dpi=200, figsize=(15,8))
        ax.set_title(filename)
        ax.plot(t[i:j],y[i:j])
        ax.set_xlabel('Time (s)')
        
    if fills != None:
        shades = [i for i in map(lambda x: (max(x[0],i),min(x[1],j)), fills)]
        if len(shades)>0:
            shades = merger(shades)
            if disp:
                for s in shades:
                    ax.axvspan(s[0]/sr, s[1]/sr, alpha=0.5, color='r')
    
    if save:
        fig.savefig(save)
        
    if len(shades)>1:
        live = map(lambda i: (shades[i][1],shades[i+1][0]), range(len(shades)-1))
    elif len(shades)==1:
        a = [i,shades[0][0],shades[0][1],j]
        live = filter(lambda x: x != None, map(lambda x: tuple(x) if x[0]!=x[1] else None,np.sort(a).reshape((int(len(a)/2),2))))
    else:
        live = [(i,j)]
    if output:
        return [i for i in live], sr, len(y)

def splitter(y_list, target, interval=1, original_sr=22050, n_mfcc=20, return_target=True, sec_bins=None):
    
    '''
    Converts a list of audio signals to mfcc bands and slices and resamples the audio signals based on 
    original_sr and interval. 
    
    Returns a 2D n x m array. n is the number of data points, with m-1 features as each mfcc band value for
    the interval or time step selected.
    '''
    
    import math
    
    # check for the shortest sample
    if not sec_bins:
        newys_lengths = []

        for i in y_list:
            newys_lengths.append(len(i))

        max_length = int(np.array(newys_lengths).min()/22050)

        # create bins that match the interval length and sampling rate
        sec_bins = []

        steps = np.arange(0, (max_length*original_sr), (original_sr*interval))

        for i in range(len(steps)-1):
            sec_bins.append([steps[i],steps[i+1]])

        # create new slices of each audio signal in y_list
        print(f'Length of time bins = {len(sec_bins)}')
    
    print('Generating slices of X.')
    
    new_X = []

    for i in sec_bins:

        for audio in y_list:

            new_X.append(np.array(audio[int(i[0]):int(i[1])]))

    new_X = np.array(new_X)
    
    print('Checking for empty arrays.')
    
    counter = 0
    
    shapes = [i.shape for i in new_X]
    clean_index = []
    
    for i in range(len(shapes)):
        if shapes[i] == (0,):
            counter += 1
        else:
            clean_index.append(i)
            
    proceed = 'y'
    
    if counter > 0:
        proceed = eval(input(f'Number of empty arrays is {counter}. Proceed and delete these arrays? (y/n?)'))
    else:
        print('No empty arrays.\n')
    
    if proceed == 'y':

        new_X = new_X[clean_index]
        
    print('Slicing original audio files.')
    
    new_Xs = []
    counter = 0
    for i in range(len(new_X)):
        if counter%500==0:
            clear_output()
            print(counter)
        new_Xs.append(librosa.feature.mfcc(new_X[i], sr=original_sr, n_mfcc=n_mfcc))
        counter+=1

    new_Xs = np.array(new_Xs)

    try:
        new_Xs = new_Xs.reshape(len(new_X), (int(math.ceil((original_sr*interval)/512)) * n_mfcc))
    except:
        print(new_Xs.shape)
        print(len(new_Xs))
        
    if return_target:
        # generate new targets    
        try:
            new_target = np.array(list(target.to_numpy()) * len(sec_bins))
        except AttributeError:
            new_target = np.array(list(target) * len(sec_bins))

        return new_Xs, new_target, sec_bins
    else:
        return new_Xs

def new_prediction(filepath):
    y, sr = librosa.load(filepath)
    silences, _, _ = shade_silence(filepath, output=True, disp=False)
    newy = []
    for j in range(len(silences)):
        if silences[j][0] != silences[j][1]:
            newy.extend(y[silences[j][0]:silences[j][1]])
    X = splitter([newy,], None, original_sr=sr, interval=0.1, n_mfcc=40, return_target=False, sec_bins=sec_bins)
    preds = model.predict(X)
    total = sum([i[1] for i in Counter(preds).most_common()])
    percent = [str(round(i[1]/total*100, 2)) + '%' for i in Counter(preds).most_common()]
    countries = {1:'American', 2:'Chinese', 3:'Korean', 4:'British', 5:'Australian', 6:'Irish'}
    return dict(zip([countries[i[0]] for i in Counter(preds).most_common()], percent))

with open('sec_bins.pkl', 'rb') as f:
    sec_bins = pickle.load(f)

with open('xgboostmodel_61.pkl', 'rb') as f:
    model = pickle.load(f)
 
os.makedirs('uploads', exist_ok=True)

@app.route('/static/<string:path>')
def webpage(path):
    return send_from_directory('', path)

@app.route('/')
def index():
    return webpage('accent.html')

@app.route('/upload', methods=['POST']) 
def upload():
    file = glob.glob('uploads/*')
    for i in file:
        os.remove(i)
    file_ = request.files['upload'] 
    filename = file_.filename 
    file_.save(f'uploads/{secure_filename(filename)}')
    return webpage('accent_submit.html')

@app.route('/static/uploads/<string:path>')
def load_me(path):
    return send_from_directory('uploads/', path)

@app.route('/predict')
def predict():
    files = glob.glob('uploads/*')
    result = new_prediction(files[0])
    result = {'results':result, 'filepath':'static/' + files[0]}
    # result['filepath'] = 'static/' + files[0]
    return jsonify(result)
