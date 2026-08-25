MIDLINE_Y = 466  # capture-region midpoint
MATCH_THRESHOLD = 50  # pixels — max distance to match a detection to a tracked troop
DEATH_FRAMES = 3  # frames_since_seen threshold to declare an enemy dead


class TroopTracker:
    def __init__(self):
        self.troops = [] # list of dicts: {id, troop, x, y, team, frames_since_seen}
        self._next_id = 0

    def clear(self):
        self.troops = []
        self._next_id = 0

    def update(self, detections, my_crowns, enemy_crowns):
        """
        Match detections to tracked troops, add new ones, detect deaths.
        Returns list of killed enemy troop dicts (only when no crowns have fallen).
        """
        for t in self.troops:
            t["frames_since_seen"] += 1

        matched_ids = set()
        unmatched = []

        for det in detections:
            best_id = None
            best_dist = float("inf")
            for t in self.troops:
                if t["id"] in matched_ids:
                    continue
                dist = ((t["x"] - det["x"]) ** 2 + (t["y"] - det["y"]) ** 2) ** 0.5
                if dist < best_dist and dist <= MATCH_THRESHOLD:
                    best_dist = dist
                    best_id = t["id"]

            if best_id is not None:
                matched_ids.add(best_id)
                troop = next(t for t in self.troops if t["id"] == best_id)
                troop["x"] = det["x"]
                troop["y"] = det["y"]
                troop["frames_since_seen"] = 0
            else:
                unmatched.append(det)

        if my_crowns == 0 and enemy_crowns == 0:
            for det in unmatched:
                team = "enemy" if det["y"] < MIDLINE_Y else "ally"
                self.troops.append({
                    "id": self._next_id,
                    "troop": det["troop"],
                    "x": det["x"],
                    "y": det["y"],
                    "team": team,
                    "frames_since_seen": 0,
                })
                self._next_id += 1

        kills = [t for t in self.troops if t["team"] == "enemy" and t["frames_since_seen"] >= DEATH_FRAMES]
        self.troops = [t for t in self.troops if t["frames_since_seen"] < DEATH_FRAMES]
        return kills
