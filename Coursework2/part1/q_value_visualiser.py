import cv2
import numpy as np

from env import Environment


class QValueVisualiser:

    def __init__(self, environment, magnification=500):
        self.environment = environment
        self.magnification = magnification
        self.half_cell_length = 0.05 * self.magnification
        # Create the initial q values image
        self.q_values_image = np.zeros([int(self.magnification), int(self.magnification), 3], dtype=np.uint8)

    def draw_q_values(self, q_values):
        # Create an empty image
        self.q_values_image.fill(0)
        # Loop over the grid cells and actions, and draw each q value
        for col in range(10):
            for row in range(10):
                # Find the q value ranges for this state
                max_q_value = np.max(q_values[col, row])
                min_q_value = np.min(q_values[col, row])
                q_value_range = max_q_value - min_q_value
                # Draw the q values for this state
                for action in range(4):
                    # Normalise the q value with respect to the minimum and maximum q values
                    q_value_norm = (q_values[col, row, action] - min_q_value) / q_value_range
                    # Draw this q value
                    x = (col / 10.0) + 0.05
                    y = (row / 10.0) + 0.05
                    # if x<0.7 and y<0.5 and y>0.4:
                    #     pass
                    # else:
                    self._draw_q_value(x, y, action, float(q_value_norm))
        # Draw the grid cells
        self._draw_grid_cells()
        # Show the image
        cv2.imshow("Q Values", self.q_values_image)
        cv2.waitKey(1)

    def draw_relative_q_values(self, q_values):
        # Create an empty image
        self.q_values_image.fill(0)
        # Find the q value ranges for this state
        max_q_value = np.max(q_values)
        min_q_value = np.min(q_values)
        q_value_range = max_q_value - min_q_value
        # Loop over the grid cells and actions, and draw each q value
        for col in range(10):
            for row in range(10):
                # Draw the q values for this state
                for action in range(4):
                    # Normalise the q value with respect to the minimum and maximum q values
                    q_value_norm = (q_values[col, row, action] - min_q_value) / q_value_range
                    # Draw this q value
                    x = (col / 10.0) + 0.05
                    y = (row / 10.0) + 0.05
                    if x<0.7 and y<0.5 and y>0.4:
                        pass
                    else:
                        self._draw_q_value(x, y, action, float(q_value_norm))
        # Draw the grid cells
        self._draw_grid_cells()
        # Show the image
        cv2.imshow("Q Values", self.q_values_image)
        cv2.waitKey(1)

    def _draw_q_value(self, x, y, action, q_value_norm):
        # First, convert state space to image space for the "up-down" axis, because the world space origin is the bottom left, whereas the image space origin is the top left
        y = 1 - y
        # Compute the image coordinates of the centre of the triangle for this action
        centre_x = x * self.magnification
        centre_y = y * self.magnification
        # Compute the colour for this q value
        colour_r = int((1 - q_value_norm) * 255)
        colour_g = int(q_value_norm * 255)
        colour_b = 0
        colour = (colour_b, colour_g, colour_r)
        # Depending on the particular action, the triangle representing the action will be drawn in a different position on the image
        if action == 0:  # Move right
            point_1_x = centre_x + self.half_cell_length
            point_1_y = centre_y + self.half_cell_length
            point_2_x = point_1_x
            point_2_y = centre_y - self.half_cell_length
            points = np.array([[centre_x, centre_y], [point_1_x, point_1_y], [point_2_x, point_2_y]], dtype=np.int32)
            cv2.fillConvexPoly(self.q_values_image, points, colour)
            cv2.polylines(self.q_values_image, [points], True, (0, 0, 0), thickness=2, lineType=cv2.LINE_AA)
        elif action == 2:  # Move up
            point_1_x = centre_x + self.half_cell_length
            point_1_y = centre_y - self.half_cell_length
            point_2_x = centre_x - self.half_cell_length
            point_2_y = point_1_y
            points = np.array([[centre_x, centre_y], [point_1_x, point_1_y], [point_2_x, point_2_y]], dtype=np.int32)
            cv2.fillConvexPoly(self.q_values_image, points, colour)
            cv2.polylines(self.q_values_image, [points], True, (0, 0, 0), thickness=2, lineType=cv2.LINE_AA)
        elif action == 1:  # Move left
            point_1_x = centre_x - self.half_cell_length
            point_1_y = centre_y - self.half_cell_length
            point_2_x = point_1_x
            point_2_y = centre_y + self.half_cell_length
            points = np.array([[centre_x, centre_y], [point_1_x, point_1_y], [point_2_x, point_2_y]], dtype=np.int32)
            cv2.fillConvexPoly(self.q_values_image, points, colour)
            cv2.polylines(self.q_values_image, [points], True, (0, 0, 0), thickness=2, lineType=cv2.LINE_AA)
        elif action == 3:  # Move down
            point_1_x = centre_x - self.half_cell_length
            point_1_y = centre_y + self.half_cell_length
            point_2_x = centre_x + self.half_cell_length
            point_2_y = point_1_y
            points = np.array([[centre_x, centre_y], [point_1_x, point_1_y], [point_2_x, point_2_y]], dtype=np.int32)
            cv2.fillConvexPoly(self.q_values_image, points, colour)
            cv2.polylines(self.q_values_image, [points], True, (0, 0, 0), thickness=2, lineType=cv2.LINE_AA)

    def _draw_grid_cells(self):
        # Draw the state cell borders
        for col in range(11):
            point_1 = (int((col / 10.0) * self.magnification), 0)
            point_2 = (int((col / 10.0) * self.magnification), int(self.magnification))
            cv2.line(self.q_values_image, point_1, point_2, (255, 255, 255), thickness=4, lineType=cv2.LINE_AA)
        for row in range(11):
            point_1 = (0, int((row / 10.0) * self.magnification))
            point_2 = (int(self.magnification), int((row / 10.0) * self.magnification))
            cv2.line(self.q_values_image, point_1, point_2, (255, 255, 255), thickness=4, lineType=cv2.LINE_AA)

    def get_image(self):
        return self.q_values_image


# Main entry point
if __name__ == "__main__":

    # Create some random q values
    q_values = np.random.uniform(0, 1, [10, 10, 4])
    # Create an environment
    environment = Environment(display=False, magnification=500)
    # Create a visualiser
    visualiser = QValueVisualiser(environment=environment, magnification=500)
    # Draw the image
    visualiser.draw_q_values(q_values)
