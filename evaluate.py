import pandas as pd
import numpy as np

from sklearn.cluster import KMeans

# class genreList:
#     def __init__(self, genreList, type):
#         self.counts = {}
#         for genre in genreList:
#             if type == 'count':
#                 self.counts[genre] = 0
#             if type == 'song':
#                 self.counts[genre] = []
    
#     def printCounts(self):
#         for genre, count in self.counts.items():
#             print(f"{genre}: {count}")

# class records:
#     def __init__(self, genreList):
#         self.records = {
#             'majority_genre': None,
#             'songs': set(),
#             'non_majority_songs': set()
#         }
#         self.genreCounts = genreCounts(genreList)

#     def add_song

# can maybe change this to a general get_clustering_object function
def get_kmeans(all_ftrs, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++')
    kmeans.fit(all_ftrs)
    return kmeans

def evaluate(n_clusters, audioFtrs_df, all_ftrs):
    # find genre with majority of songs in cluster
    # find songs in cluster
    # find songs not corresponding to majority in cluster
    
    cluster_indices = [[] for i in range(n_clusters)]

    kmeans = get_kmeans(all_ftrs, n_clusters)

    for i,label in enumerate(kmeans.labels_):
        cluster_indices[label].append(i)
    
    all_records = []
    all_counts = []
    for cluster in cluster_indices:
        counts = {'dubstep': 0, 'house': 0, 'dnb': 0, 'rock': 0, 'rap': 0}

        records = {
        'majority': None, 
        'songs': {'dubstep': [], 'house': [], 'dnb': [], 'rock': [], 'rap': []}
        }

        for ind in cluster:
            row = audioFtrs_df.iloc[ind]
            song_name = row['name']
            genre_name = row['genre']
            counts[genre_name] += 1
            records['songs'][genre_name].append(song_name)
        
        majority_count = 0
        majority_genre = None

        for genre, songs in records['songs'].items():
            if majority_count < len(songs):
                majority_count = len(songs)
                majority_genre = genre
        
        records['majority'] = majority_genre

        all_records.append(records)
        all_counts.append(counts)

    return all_records, all_counts, kmeans


