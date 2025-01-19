import numpy as np
import heapq

def uniform_cost_search(graph, start_image, reference_image=0):
    """
    Perform Uniform Cost Search (UCS) on the given graph.

    Parameters:
        graph (dict): The graph represented as {(number_of_image_1, number_of_image_2): (homography, step_cost)}.
        start_image (int): The starting image number.
        reference_image (int): The reference image number (default: 0).

    Returns:
        path (tuple): Sequence of image numbers leading to the reference image.
        total_cost (float): The total cost of the path.
    """

    # Priority queue: (cumulative_cost, current_image, path_so_far)
    pq = []
    heapq.heappush(pq, (0, start_image, [start_image]))

    # Dictionary to track the best cost for nodes in the queue
    in_queue = {start_image: 0}

    # Visited set to store the lowest cost at which a node was reached
    visited = {}

    while pq:
        # Pop the node with the lowest cumulative cost
        cumulative_cost, current_image, path_so_far = heapq.heappop(pq)

        # If this node has been visited with a lower cost, skip it
        if current_image in visited and visited[current_image] <= cumulative_cost:
            continue

        # Remove the node from the queue tracking
        in_queue.pop(current_image, None)

        # Mark the current node as visited
        visited[current_image] = cumulative_cost

        # If the current image is the reference image, return the path
        if current_image == reference_image:
            return tuple(path_so_far), cumulative_cost

        # Expand the current node by iterating through the graph
        for (to_image, from_image), (homography, step_cost) in graph.items():
            if from_image == current_image:
                # Calculate the new cumulative cost
                new_cost = cumulative_cost + step_cost

                # If the node is already in the visited set with a lower cost, skip it
                if to_image in visited and visited[to_image] <= new_cost:
                    continue

                # If the node is in the queue and the new cost is better, update it
                if to_image in in_queue and in_queue[to_image] > new_cost:
                    in_queue[to_image] = new_cost
                    heapq.heappush(pq, (new_cost, to_image, path_so_far + [to_image]))

                # If the node is not in the queue, add it
                elif to_image not in in_queue:
                    in_queue[to_image] = new_cost
                    heapq.heappush(pq, (new_cost, to_image, path_so_far + [to_image]))

    # If the queue is empty and we didn't reach the reference image
    return None, float('inf')