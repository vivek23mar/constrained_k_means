from flask import jsonify, request
from k_means_constrained import KMeansConstrained
import numpy as np
from . import app

@app.route('/calculate-kmeans', methods=['POST'])
def calculate_kmeans():
    try:
        # Parse JSON data from request
        body = request.get_json()
        raw_data = body['data']
        n_clusters = body['n_clusters']
        size_min = body.get('size_min', 1)
        size_max = body.get('size_max', len(raw_data))

        # Extract coordinates for clustering
        coordinates = [[item['lat'], item['lng']] for item in raw_data]

        # Perform constrained K-means clustering
        kmeans = KMeansConstrained(
            n_clusters=n_clusters,
            size_min=size_min,
            size_max=size_max,
            random_state=42
        )
        kmeans.fit(coordinates)

        # Map results back to the original data
        clustered_data = [
            {
                "orderId": raw_data[i]['orderId'],
                "orderDistance": raw_data[i]['orderDistance'],
                "orderETA": raw_data[i]['orderETA'],
                "paymentMode": raw_data[i]['paymentMode'],
                "storeCoordinates": raw_data[i]['storeCoordinates'],
                "totalPrice": raw_data[i]['totalPrice'],
                "userLocation": raw_data[i]['userLocation'],
                "lat": raw_data[i]['lat'],
                "lng": raw_data[i]['lng'],
                "cluster": int(kmeans.labels_[i])
            }
            for i in range(len(raw_data))
        ]

        # Prepare response
        response = {
            # "centroids": kmeans.cluster_centers_.tolist(),
            "clusters": clustered_data
        }
        return jsonify(response), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 400
