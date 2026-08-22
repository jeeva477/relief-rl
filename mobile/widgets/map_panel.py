from kivy.graphics import Color, Line, Rectangle
from kivy.properties import ListProperty
from kivy.uix.widget import Widget


class MapPanel(Widget):
    """Lightweight offline route visualization used when a native map SDK is unavailable."""

    route_points = ListProperty([])

    def set_route(self, route):
        points = []
        for segment in route or []:
            coordinates = segment.get("coordinates", []) if isinstance(segment, dict) else []
            for point in coordinates:
                if isinstance(point, dict) and "latitude" in point and "longitude" in point:
                    points.append(point)
        self.route_points = points
        self._redraw()

    def on_size(self, *args):
        self._redraw()

    def on_pos(self, *args):
        self._redraw()

    def _redraw(self):
        self.canvas.after.clear()
        with self.canvas.after:
            Color(0.08, 0.10, 0.14, 1)
            Rectangle(pos=self.pos, size=self.size)
            if len(self.route_points) < 2:
                return
            xs = [float(p["longitude"]) for p in self.route_points]
            ys = [float(p["latitude"]) for p in self.route_points]
            x_span = max(xs) - min(xs)
            y_span = max(ys) - min(ys)
            if x_span == 0 and y_span == 0:
                return
            x_scale = self.width * 0.8 / x_span if x_span else float("inf")
            y_scale = self.height * 0.8 / y_span if y_span else float("inf")
            scale = min(x_scale, y_scale)
            pts = []
            for x, y in zip(xs, ys):
                pts += [self.x + self.width * 0.1 + (x - min(xs)) * scale,
                        self.y + self.height * 0.1 + (y - min(ys)) * scale]
            Color(0.2, 0.8, 0.4, 1)
            Line(points=pts, width=3)
