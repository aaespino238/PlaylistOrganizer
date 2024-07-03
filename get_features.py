import pandas as pd
import numpy as np
from tqdm import tqdm

"""
easy manipulation and selection of features
allow for variation in number of clusters and number of sections considered 

visualization of effectiveness 

number of songs corresponding to each genre per cluster
list of songs in each cluster


IDEA FOR TOMORROW: 
- try to allow for easy selection of both songs and sections corresponding to mood
- first try to see how well the loudest section strategy works
"""

def get_audio_features(audio_ftrs_df, irrelevant_ftrs):
    """
    audio_ftrs_df: the entire audio_features df obtained through spotify api
    relevant_ftrs: the desired features for clustering
    """
    audio_ftrs = pd.DataFrame(audio_ftrs_df, copy=True)
    audio_ftrs = audio_ftrs.drop(irrelevant_ftrs, axis=1)
    return audio_ftrs

def get_n_loudest_sections(song_sections, n):
    """
    song_sections: sections of a song obtained through spotify api
                n: number of sections desired
    """
    loudnessList = []
    for section in song_sections:
        loudnessList.append(section['loudness'])
    
    loudestIndices = np.argsort(loudnessList)[:n]

    return song_sections[loudestIndices]

def get_section_segments(section, segments):
    """
    section: the section in consideration
    segments: the segments corresponding to the song in consideration
    """
    start_time = section['start'] 
    end_time = start_time + section['duration'] 

    section_segments = []
    for seg in segments:
        if seg['start'] >= start_time and seg['start'] + seg['duration'] <= end_time:
            section_segments.append(seg)
    
    return section_segments

def get_section_mean_timbre(section_segments):
    timbres = [seg['timbre'] for seg in section_segments]
    if timbres:
        out = np.mean(timbres, axis=0)
        return out
    else:
        return None

def get_all_features(audio_features, relevant_ftrs, audio_analysis, num_sections):
    audio_ftrs = get_audio_features(audio_features, relevant_ftrs)

    all_segments = [np.array(song['segments']) for song in audio_analysis]
    all_sections = [np.array(song['sections']) for song in audio_analysis]
    
    all_timbres = []

    # for each song, obtain mean timbres for each of the n loudest sections 
    bad_indices = []
    for i, song_sections in tqdm(enumerate(all_sections)):
        loudest_sections = get_n_loudest_sections(song_sections, num_sections)
        
        song_segments = all_segments[i]
        curr_timbres = []
       
        for section in loudest_sections:
            section_segments = get_section_segments(section, song_segments)
            curr_timbres.append(get_section_mean_timbre(section_segments))
        
        badIndex = False
        for t in curr_timbres:
            if not np.any(t):
                bad_indices.append(i)
                badIndex = True
        
        if not badIndex:
            all_timbres.append(np.concat(curr_timbres))

    audio_ftrs.drop(audio_ftrs.index[bad_indices], inplace=True)
    audio_ftrs.reset_index(drop=True, inplace=True)

    timbre_ftrs = pd.DataFrame(all_timbres)
    all_ftrs = pd.concat([audio_ftrs, timbre_ftrs], axis=1)

    cleaned_audio_features = audio_features.drop(audio_features.index[bad_indices])
    cleaned_audio_features.reset_index(drop=True, inplace=True)

    return cleaned_audio_features, all_ftrs







