response_schema = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "start_time": {
                    "type": "string",
                    "description":"format the time in the format MM:SS"
                },
                "end_time": {
                    "type": "string",
                    "description":"format the time in the format MM:SS"
                },
                "speaker": {"type": "string"},
                "utterance": {
                    "type": "string", 
                    "description": "transcription of whatever was spoken by the speaker in the time interval"
                },
                "loudness": {
                    "type": "number", 
                    "format": "double", 
                    "description": "average loudness of the speaker's voice between the start_time and end_time timestamps in dBFS scale."
                },
                "sentiment": {
                    "type": "number",
                    "format": "double",
                    "description": "average sentiment polarity of the speaker for the utterance between the start_time and end_time timestamps."
                }
            },
            "required": ["start_time", "end_time", "speaker", "utterance", "loudness", "sentiment"],
        },
    }