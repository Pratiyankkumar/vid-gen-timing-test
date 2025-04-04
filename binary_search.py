from manim import *

class BinarySearchVisualization(Scene):
    def construct(self):
        # Title
        title = Text("Binary Search Algorithm").scale(0.8).to_edge(UP)
        self.play(Write(title))
        
        # Create array
        values = [10, 20, 30, 40, 50, 60, 70, 80, 90]
        squares = VGroup()
        labels = VGroup()
        
        for i, value in enumerate(values):
            square = Square(side_length=0.8).set_stroke(WHITE, 2)
            square.set_fill(BLUE, opacity=0.3)
            label = Text(str(value), font_size=24)
            square.move_to([i - len(values)/2 + 0.5, 0, 0])
            label.move_to(square.get_center())
            squares.add(square)
            labels.add(label)
        
        array = VGroup(squares, labels)
        self.play(Create(squares), FadeIn(labels))
        self.wait()
        
        # Target value
        target_text = Text("Find: 50", font_size=28).to_edge(DOWN)
        self.play(Write(target_text))
        self.wait()
        
        # Binary search animation
        low = 0
        high = len(values) - 1
        
        low_arrow = Arrow(start=DOWN, end=UP).next_to(squares[low], DOWN)
        low_label = Text("Low", font_size=20).next_to(low_arrow, DOWN)
        high_arrow = Arrow(start=DOWN, end=UP).next_to(squares[high], DOWN)
        high_label = Text("High", font_size=20).next_to(high_arrow, DOWN)
        
        self.play(Create(low_arrow), Write(low_label),
                  Create(high_arrow), Write(high_label))
        self.wait()
        
        found = False
        steps = 0
        
        # We'll loop until we find the value or exhaust the search space
        while low <= high and steps < 5:  # Limit steps to prevent infinite loop
            mid = (low + high) // 2
            
            # Highlight mid
            mid_arrow = Arrow(start=UP, end=DOWN).next_to(squares[mid], UP)
            mid_label = Text("Mid", font_size=20).next_to(mid_arrow, UP)
            squares[mid].set_fill(YELLOW, opacity=0.7)
            
            self.play(Create(mid_arrow), Write(mid_label))
            self.wait()
            
            # Check if mid is the target
            if values[mid] == 50:  # Our target value
                found = True
                squares[mid].set_fill(GREEN, opacity=0.7)
                found_text = Text("Found at index " + str(mid), font_size=28).to_edge(DOWN)
                self.play(ReplacementTransform(target_text, found_text))
                self.wait()
                break
                
            elif values[mid] < 50:  # Target is to the right
                squares[mid].set_fill(RED, opacity=0.3)
                self.play(FadeOut(low_arrow), FadeOut(low_label))
                low = mid + 1
                low_arrow = Arrow(start=DOWN, end=UP).next_to(squares[low], DOWN)
                low_label = Text("Low", font_size=20).next_to(low_arrow, DOWN)
                self.play(Create(low_arrow), Write(low_label))
            
            else:  # Target is to the left
                squares[mid].set_fill(RED, opacity=0.3)
                self.play(FadeOut(high_arrow), FadeOut(high_label))
                high = mid - 1
                high_arrow = Arrow(start=DOWN, end=UP).next_to(squares[high], DOWN)
                high_label = Text("High", font_size=20).next_to(high_arrow, DOWN)
                self.play(Create(high_arrow), Write(high_label))
            
            # Clean up mid highlight for next iteration
            self.play(FadeOut(mid_arrow), FadeOut(mid_label))
            steps += 1
            self.wait()
        
        if not found:
            not_found = Text("Value not found", font_size=28).to_edge(DOWN)
            self.play(ReplacementTransform(target_text, not_found))
        
        self.wait(2)


class SortingAlgorithm(Scene):
    def construct(self):
        # Title
        title = Text("Bubble Sort").scale(0.8).to_edge(UP)
        self.play(Write(title))
        self.wait()
        
        # Create array
        values = [7, 2, 9, 4, 1, 8, 5, 3, 6]
        rectangles = VGroup()
        labels = VGroup()
        
        for i, value in enumerate(values):
            height = value * 0.4  # Scale the height based on value
            rect = Rectangle(width=0.8, height=height).set_stroke(WHITE, 2)
            rect.set_fill(BLUE, opacity=0.5)
            rect.move_to([i - len(values)/2 + 0.5, height/2 - 1, 0])
            label = Text(str(value), font_size=24).move_to(rect.get_center())
            rectangles.add(rect)
            labels.add(label)
        
        array = VGroup(rectangles, labels)
        self.play(Create(rectangles), FadeIn(labels))
        self.wait()
        
        # Bubble sort animation
        sorted_values = values.copy()
        
        for i in range(len(sorted_values)):
            swapped = False
            for j in range(len(sorted_values) - i - 1):
                # Highlight current pair being compared
                rectangles[j].set_fill(YELLOW, opacity=0.8)
                rectangles[j+1].set_fill(YELLOW, opacity=0.8)
                self.wait(0.5)
                
                if sorted_values[j] > sorted_values[j+1]:
                    # Swap values
                    sorted_values[j], sorted_values[j+1] = sorted_values[j+1], sorted_values[j]
                    
                    # Swap rectangles and labels
                    rect1 = rectangles[j]
                    rect2 = rectangles[j+1]
                    label1 = labels[j]
                    label2 = labels[j+1]
                    
                    # Animate swap
                    self.play(
                        rect1.animate.move_to([j+1 - len(values)/2 + 0.5, rect1.get_center()[1], 0]),
                        rect2.animate.move_to([j - len(values)/2 + 0.5, rect2.get_center()[1], 0]),
                        label1.animate.move_to([j+1 - len(values)/2 + 0.5, label1.get_center()[1], 0]),
                        label2.animate.move_to([j - len(values)/2 + 0.5, label2.get_center()[1], 0])
                    )
                    
                    # Update the VGroups
                    rectangles[j], rectangles[j+1] = rectangles[j+1], rectangles[j]
                    labels[j], labels[j+1] = labels[j+1], labels[j]
                    swapped = True
                
                # Reset highlight
                rectangles[j].set_fill(BLUE, opacity=0.5)
                if j < len(rectangles) - i - 1:
                    rectangles[j+1].set_fill(BLUE, opacity=0.5)
                else:
                    # The last element in this pass is now sorted
                    rectangles[j+1].set_fill(GREEN, opacity=0.7)
            
            if not swapped:
                # If no swaps were made, the array is sorted
                for k in range(len(rectangles) - i):
                    rectangles[k].set_fill(GREEN, opacity=0.7)
                break
        
        # All sorted
        sorted_text = Text("Sorted!", font_size=28).to_edge(DOWN)
        self.play(Write(sorted_text))
        self.wait(2)


class GraphTraversal(Scene):
    def construct(self):
        # Title
        title = Text("Breadth-First Search (BFS)").scale(0.8).to_edge(UP)
        self.play(Write(title))
        self.wait()
        
        # Create graph nodes
        nodes = {}
        positions = {
            'A': [-3, 1, 0],
            'B': [-1, 2, 0],
            'C': [-1, 0, 0],
            'D': [1, 2, 0],
            'E': [1, 0, 0],
            'F': [3, 1, 0],
        }
        
        # Adjacency list
        adjacency = {
            'A': ['B', 'C'],
            'B': ['A', 'D'],
            'C': ['A', 'E'],
            'D': ['B', 'F'],
            'E': ['C', 'F'],
            'F': ['D', 'E'],
        }
        
        # Create circles for nodes
        for node, pos in positions.items():
            circle = Circle(radius=0.5).set_stroke(WHITE, 2)
            circle.set_fill(BLUE, opacity=0.5)
            circle.move_to(pos)
            label = Text(node, font_size=24).move_to(circle.get_center())
            nodes[node] = VGroup(circle, label)
            self.play(Create(circle), Write(label), run_time=0.5)
        
        # Create edges
        edges = VGroup()
        for node, neighbors in adjacency.items():
            for neighbor in neighbors:
                if neighbor > node:  # To avoid creating the same edge twice
                    edge = Line(
                        nodes[node][0].get_center(),
                        nodes[neighbor][0].get_center()
                    ).set_stroke(WHITE, 2)
                    edges.add(edge)
        
        self.play(Create(edges))
        self.wait()
        
        # BFS animation
        start_node = 'A'
        queue = [start_node]
        visited = [start_node]
        
        # Highlight start node
        nodes[start_node][0].set_fill(GREEN, opacity=0.8)
        self.wait()
        
        # Display queue
        queue_text = Text("Queue: " + str(queue), font_size=20).to_edge(DOWN)
        self.play(Write(queue_text))
        
        # Process nodes using BFS
        while queue:
            # Dequeue a node
            current = queue.pop(0)
            current_queue = queue.copy()
            
            # Update queue display
            new_queue_text = Text("Queue: " + str(current_queue), font_size=20).to_edge(DOWN)
            self.play(ReplacementTransform(queue_text, new_queue_text))
            queue_text = new_queue_text
            
            # Process current node
            self.play(
                nodes[current][0].animate.set_fill(YELLOW, opacity=0.8)
            )
            self.wait(0.5)
            
            # Add unvisited neighbors to queue
            for neighbor in adjacency[current]:
                if neighbor not in visited:
                    # Highlight edge being traversed
                    edge_to_highlight = None
                    for edge in edges:
                        if (np.array_equal(edge.get_start(), nodes[current][0].get_center()) and 
                            np.array_equal(edge.get_end(), nodes[neighbor][0].get_center())) or \
                            (np.array_equal(edge.get_start(), nodes[neighbor][0].get_center()) and 
                            np.array_equal(edge.get_end(), nodes[current][0].get_center())):
                            edge_to_highlight = edge
                            break
                    
                    if edge_to_highlight:
                        self.play(
                            edge_to_highlight.animate.set_stroke(color=YELLOW, width=4)
                        )
                    
                    # Mark neighbor as visited and add to queue
                    visited.append(neighbor)
                    queue.append(neighbor)
                    
                    # Highlight neighbor as queued
                    self.play(
                        nodes[neighbor][0].animate.set_fill(PINK, opacity=0.8)
                    )
                    
                    # Update queue display
                    new_queue_text = Text("Queue: " + str(queue), font_size=20).to_edge(DOWN)
                    self.play(ReplacementTransform(queue_text, new_queue_text))
                    queue_text = new_queue_text
            
            # Mark current node as processed
            self.play(
                nodes[current][0].animate.set_fill(GREEN, opacity=0.8)
            )
            self.wait(0.5)
        
        # Display traversal order
        traversal_text = Text("BFS Traversal: " + " → ".join(visited), font_size=24)
        traversal_text.to_edge(DOWN)
        self.play(ReplacementTransform(queue_text, traversal_text))
        
        self.wait(2)


# Add a new class for Depth-First Search for comparison
class DFSTraversal(Scene):
    def construct(self):
        # Title
        title = Text("Depth-First Search (DFS)").scale(0.8).to_edge(UP)
        self.play(Write(title))
        self.wait()
        
        # Create graph nodes - same graph as BFS for comparison
        nodes = {}
        positions = {
            'A': [-3, 1, 0],
            'B': [-1, 2, 0],
            'C': [-1, 0, 0],
            'D': [1, 2, 0],
            'E': [1, 0, 0],
            'F': [3, 1, 0],
        }
        
        # Adjacency list
        adjacency = {
            'A': ['B', 'C'],
            'B': ['A', 'D'],
            'C': ['A', 'E'],
            'D': ['B', 'F'],
            'E': ['C', 'F'],
            'F': ['D', 'E'],
        }
        
        # Create circles for nodes
        for node, pos in positions.items():
            circle = Circle(radius=0.5).set_stroke(WHITE, 2)
            circle.set_fill(BLUE, opacity=0.5)
            circle.move_to(pos)
            label = Text(node, font_size=24).move_to(circle.get_center())
            nodes[node] = VGroup(circle, label)
            self.play(Create(circle), Write(label), run_time=0.5)
        
        # Create edges
        edges = VGroup()
        for node, neighbors in adjacency.items():
            for neighbor in neighbors:
                if neighbor > node:  # To avoid creating the same edge twice
                    edge = Line(
                        nodes[node][0].get_center(),
                        nodes[neighbor][0].get_center()
                    ).set_stroke(WHITE, 2)
                    edges.add(edge)
        
        self.play(Create(edges))
        self.wait()
        
        # DFS animation
        start_node = 'A'
        stack = [start_node]
        visited = []
        
        # Display stack
        stack_text = Text("Stack: " + str(stack), font_size=20).to_edge(DOWN)
        self.play(Write(stack_text))
        
        # Process nodes using DFS
        while stack:
            # Pop a node from stack
            current = stack.pop()
            current_stack = stack.copy()
            
            # Update stack display
            new_stack_text = Text("Stack: " + str(current_stack), font_size=20).to_edge(DOWN)
            self.play(ReplacementTransform(stack_text, new_stack_text))
            stack_text = new_stack_text
            
            if current not in visited:
                # Process current node
                visited.append(current)
                self.play(
                    nodes[current][0].animate.set_fill(YELLOW, opacity=0.8)
                )
                self.wait(0.5)
                
                # Add unvisited neighbors to stack in reverse order (to process in the right order)
                neighbors_to_add = [n for n in reversed(adjacency[current]) if n not in visited]
                for neighbor in neighbors_to_add:
                    # Highlight edge being considered
                    edge_to_highlight = None
                    for edge in edges:
                        if (np.array_equal(edge.get_start(), nodes[current][0].get_center()) and 
                            np.array_equal(edge.get_end(), nodes[neighbor][0].get_center())) or \
                            (np.array_equal(edge.get_start(), nodes[neighbor][0].get_center()) and 
                            np.array_equal(edge.get_end(), nodes[current][0].get_center())):
                            edge_to_highlight = edge
                            break
                    
                    if edge_to_highlight:
                        self.play(
                            edge_to_highlight.animate.set_stroke(color=YELLOW, width=4)
                        )
                    
                    # Add to stack
                    stack.append(neighbor)
                    
                    # Show node being added to stack
                    self.play(
                        nodes[neighbor][0].animate.set_fill(PINK, opacity=0.5)
                    )
                    
                    # Update stack display
                    new_stack_text = Text("Stack: " + str(stack), font_size=20).to_edge(DOWN)
                    self.play(ReplacementTransform(stack_text, new_stack_text))
                    stack_text = new_stack_text
                
                # Mark current node as processed
                self.play(
                    nodes[current][0].animate.set_fill(GREEN, opacity=0.8)
                )
                self.wait(0.5)
        
        # Display traversal order
        traversal_text = Text("DFS Traversal: " + " → ".join(visited), font_size=24)
        traversal_text.to_edge(DOWN)
        self.play(ReplacementTransform(stack_text, traversal_text))
        
        self.wait(2)