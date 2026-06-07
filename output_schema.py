import uuid
import json
from datetime import datetime

data = {

    "session_id": str(uuid.uuid4()),

    "question_id": "Q1",

    "timestamp": str(datetime.now()),

    "frames": [

        {
            "frame_number": 1,
            "face_detected": True,
            "face_count": 1
        },

        {
            "frame_number": 2,
            "face_detected": True,
            "face_count": 1
        },

        {
            "frame_number": 3,
            "face_detected": False,
            "face_count": 0
        }

    ]
}

print(json.dumps(data, indent=4))