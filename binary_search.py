from manim import *

class SimpleAnimation(Scene):
    def construct(self):
        # Title
        title = Text("Binary Search Algorithm", font_size=40)
        title.to_edge(UP)
        self.play(Write(title))
        
        # Create an array
        array_size = 10
        values = list(range(10, array_size*10 + 10, 10))  # [10, 20, 30, ..., 100]
        target_value = 70
        
        # Create rectangles for array elements
        squares = []
        texts = []
        
        for i in range(array_size):
            square = Square(side_length=0.7)
            square.set_stroke(WHITE, 2)
            text = Text(str(values[i]), font_size=24)
            text.move_to(square.get_center())
            
            squares.append(square)
            texts.append(text)
        
        # Arrange in a row
        vgroup = VGroup(*squares)
        vgroup.arrange(RIGHT, buff=0.1)
        vgroup.move_to(ORIGIN)
        
        # Add indices below squares
        indices = []
        for i in range(array_size):
            index = Text(str(i), font_size=16)
            index.next_to(squares[i], DOWN, buff=0.2)
            indices.append(index)
        
        # Show array with values
        self.play(
            LaggedStart(*[Create(square) for square in squares], lag_ratio=0.1),
            run_time=2
        )
        
        self.play(
            LaggedStart(*[Write(text) for text in texts], lag_ratio=0.1),
            LaggedStart(*[Write(index) for index in indices], lag_ratio=0.1),
            run_time=2
        )
        
        # Target value label
        target_text = Text(f"Target: {target_value}", font_size=32)
        target_text.to_edge(UP).shift(DOWN*0.5)
        self.play(Write(target_text))
        
        # Binary search pointers
        left_arrow = Arrow(start=DOWN, end=UP, color=BLUE)
        left_arrow.next_to(squares[0], DOWN, buff=0.5)
        left_label = Text("left", font_size=20, color=BLUE)
        left_label.next_to(left_arrow, DOWN, buff=0.1)
        
        right_arrow = Arrow(start=DOWN, end=UP, color=RED)
        right_arrow.next_to(squares[-1], DOWN, buff=0.5)
        right_label = Text("right", font_size=20, color=RED)
        right_label.next_to(right_arrow, DOWN, buff=0.1)
        
        mid_arrow = Arrow(start=DOWN, end=UP, color=GREEN)
        mid_index = (array_size - 1) // 2
        mid_arrow.next_to(squares[mid_index], DOWN, buff=0.5)
        mid_label = Text("mid", font_size=20, color=GREEN)
        mid_label.next_to(mid_arrow, DOWN, buff=0.1)
        
        # Show pointers
        self.play(
            Create(left_arrow), Write(left_label),
            Create(right_arrow), Write(right_label),
            run_time=1
        )
        
        # Algorithm steps
        left, right = 0, array_size - 1
        found = False
        steps = []
        
        while left <= right:
            mid = (left + right) // 2
            
            # Show mid pointer
            mid_arrow_new = Arrow(start=DOWN, end=UP, color=GREEN)
            mid_arrow_new.next_to(squares[mid], DOWN, buff=0.5)
            
            self.play(
                ReplacementTransform(mid_arrow, mid_arrow_new) if 'mid_arrow' in locals() else Create(mid_arrow_new),
                Write(mid_label) if 'mid_label' not in locals() else mid_label.animate.next_to(mid_arrow_new, DOWN, buff=0.1),
                run_time=1
            )
            mid_arrow = mid_arrow_new
            
            # Highlight current element
            self.play(squares[mid].animate.set_fill(YELLOW, opacity=0.5), run_time=0.5)
            
            # Compare with target
            comparison_text = Text("", font_size=28)
            
            if values[mid] == target_value:
                comparison_text = Text(f"{values[mid]} == {target_value} (Found!)", font_size=28, color=GREEN)
                comparison_text.to_edge(DOWN, buff=1)
                found = True
                steps.append((left, right, mid, True))
                
                self.play(Write(comparison_text))
                self.play(Circumscribe(squares[mid], color=GREEN, time_width=0.5, run_time=2))
                break
                
            elif values[mid] < target_value:
                comparison_text = Text(f"{values[mid]} < {target_value} (Go right)", font_size=28, color=BLUE)
                comparison_text.to_edge(DOWN, buff=1)
                left = mid + 1
                steps.append((left, right, mid, False))
                
                # Move left pointer
                left_arrow_new = Arrow(start=DOWN, end=UP, color=BLUE)
                left_arrow_new.next_to(squares[left], DOWN, buff=0.5)
                
                self.play(Write(comparison_text))
                self.play(
                    ReplacementTransform(left_arrow, left_arrow_new),
                    left_label.animate.next_to(left_arrow_new, DOWN, buff=0.1),
                    run_time=1
                )
                left_arrow = left_arrow_new
                
            else:  # values[mid] > target_value
                comparison_text = Text(f"{values[mid]} > {target_value} (Go left)", font_size=28, color=RED)
                comparison_text.to_edge(DOWN, buff=1)
                right = mid - 1
                steps.append((left, right, mid, False))
                
                # Move right pointer
                right_arrow_new = Arrow(start=DOWN, end=UP, color=RED)
                right_arrow_new.next_to(squares[right], DOWN, buff=0.5)
                
                self.play(Write(comparison_text))
                self.play(
                    ReplacementTransform(right_arrow, right_arrow_new),
                    right_label.animate.next_to(right_arrow_new, DOWN, buff=0.1),
                    run_time=1
                )
                right_arrow = right_arrow_new
            
            # Un-highlight current element
            self.play(squares[mid].animate.set_fill(opacity=0), run_time=0.5)
            self.play(FadeOut(comparison_text), run_time=0.5)
            
            if left > right:
                not_found = Text("Element not found!", font_size=32, color=RED)
                not_found.to_edge(DOWN, buff=1)
                self.play(Write(not_found))
        
        # Summary
        if found:
            result_text = Text(f"Found {target_value} at index {mid}", font_size=32, color=GREEN)
        else:
            result_text = Text(f"{target_value} not found in the array", font_size=32, color=RED)
        
        result_text.to_edge(DOWN, buff=1)
        self.play(Write(result_text))
        
        # End
        self.wait(2)