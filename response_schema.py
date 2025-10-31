combined_components_response_schema = {
        "type": "object",
        "properties":{
            "summary":{
                "type": "object",
                "properties": {
                    "chapters": {
                        "type": "array",
                        "description": "Divide the conversation into chapters based on the sub-topics of discussion.",
                        "items": {
                            "type": "string"
                        }
                    },
                    "summary": {
                        "type": "string",
                        "description": "Brief but information dense summary of the conversation transcript provided."
                    },
                    "key_questions": {
                        "type": "array",
                        "description": "Outstanding questions or areas of uncertainty that may or may not have been raised during the meeting and may require further discussion or investigation in order to fulfill the purpose of the call.",
                        "items": {
                            "type": "string"
                        }
                    },
                    "notepad": {
                        "type": "array",
                        "description": "Any comments or observations of importance which might be of importance, but aren't captured in the summary",
                        "items": {
                            "type": "string"
                        }
                    },
                },
                "required": [
                    "chapters",
                    "summary",
                    "key_questions",
                    "notepad",
                ],
            },
            "aggregation": {
                "type": "object",
                "properties": {
                    "key_issues_discussed": {
                        "type": "array",
                        "description": "List of sub-topics discussed in the conversation based on the call transcript. Try to keep the sub-topic titles brief.",
                        "items": {"type": "string"},
                    },
                    "action_items": {
                        "type": "array",
                        "description": "List of tasks to be done by parties involved in the conversation based on the call transcript.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "assignee": {
                                    "type": "string",
                                    "description": "Role of the person in the transcript responsible for performing the action item."
                                },
                                "action_item": {
                                    "type": "string",
                                    "description": "Action to be done by the respective assignee."
                                },
                                "deadline": {
                                    "type": "string",
                                    "description": "Deadline for the action to be done by the assignee."
                                }
                            }
                        },
                    },
                },
                "required": ["key_issues_discussed", "action_items"],
            },
            "intent_sources":{
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "keyword": {
                            "type": "string"
                        },
                        "sources": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties":{
                                    "speaker": {
                                        "type": "string"
                                    },
                                    "source": {
                                        "type": "string",
                                        "description": "A list of sentences or phrases (sources) from the transcript where the keyword or its corresponding translation in other languages or semantic references appear.",
                                    }
                                }
                                
                            }
                        },
                    },
                    "required": ["keyword", "sources"],
                },
            },
            "entity_filler_count":{
                "type": "object",
                "properties": {
                    "entities": {
                        "type": "array",
                        "description": "List of entities proper nouns mentioned in the transcript",
                        "items": {
                            "type": "string",
                            "description": "name of the entity",
                        },
                    },
                    "filler_words": {
                        "type": "array",
                        "description": "List of fillers used in the conversation according to the transcript.",
                        "items":{
                            "type": "string",
                            "description": "filler word present in the transcript"
                        }
                    },
                },
                "required": ["entities", "filler_words"],
            },
   
            "customer_insights":{
                "type": "object",
                "properties": {
                    "customer_insights": {
                        "type": "object",
                        "properties": {
                            "personal_information": {
                                "type": "object",
                                "properties": {
                                    "name": {
                                        "type": "object",
                                        "properties": {
                                            "first_name": {"type": "string"},
                                            "last_name": {"type": "string"},
                                        },
                                    },
                                    "contact": {
                                        "type": "object",
                                        "properties": {
                                            "phone_number": {"type": "string"},
                                            "email": {"type": "string"},
                                        },
                                    },
                                    "address": {
                                        "type": "object",
                                        "properties": {
                                            "city": {"type": "string"},
                                            "state": {"type": "string"},
                                            "postal_code": {"type": "string"},
                                            "country": {"type": "string"},
                                        },
                                    },
                                    "company": {
                                        "type": "object",
                                        "properties": {
                                            "company_name": {"type": "string"},
                                        },
                                    },
                                },
                                "required": ["name", "contact", "address", "company"],
                            },
                            "customer_issue": {
                                "type": "object",
                                "properties": {
                                    "primary_concern": {"type": "string"},
                                    "issue_details": {"type": "string"},
                                    "urgency_level": {
                                        "type": "string",
                                        "enum": ["low", "medium", "high"],
                                    },
                                },
                                "required": [
                                    "primary_concern",
                                    "issue_details",
                                    "urgency_level",
                                ],
                            },
                            "customer_feedback": {
                                "type": "object",
                                "properties": {
                                    "explicit_feedback": {"type": "string"},
                                    "satisfaction_level": {
                                        "type": "string",
                                        "enum": ["satisfied", "neutral", "dissatisfied"],
                                    },
                                },
                                "required": ["explicit_feedback", "satisfaction_level"],
                            },
                            "crm_fields": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                    "field_name": {"type": "string"},
                                    "value": {"type": "string"},
                                    "confidence": {"type": "number"},
                                    "evidence": {"type": "string"}
                                    },
                                    "required": ["field_name", "value", "confidence", "evidence"]
                                }
                            }

                        },
                        "required": [
                            "personal_information",
                            "customer_issue",
                            "customer_feedback",
                        ],
                    }
                },
                "required": ["customer_insights"],
            }
        },
        "required":[
            "summary",
            "aggregation",
            "intent_sources",
            "entity_filler_count",
            "boolean_checks",
            "customer_insights"
        ]
    }

combined_components_1_response_schema = {
        "type": "object",
        "properties":{
            "summary":{
                "type": "object",
                "properties": {
                    "chapters": {
                        "type": "array",
                        "description": "Divide the conversation into chapters based on the sub-topics of discussion.",
                        "items": {
                            "type": "string"
                        }
                    },
                    "summary": {
                        "type": "string",
                        "description": "Brief but information dense summary of the conversation transcript provided."
                    },
                    "key_questions": {
                        "type": "array",
                        "description": "Outstanding questions or areas of uncertainty that may or may not have been raised during the meeting and may require further discussion or investigation in order to fulfill the purpose of the call.",
                        "items": {
                            "type": "string"
                        }
                    },
                    "notepad": {
                        "type": "array",
                        "description": "Any comments or observations of importance which might be of importance, but aren't captured in the summary",
                        "items": {
                            "type": "string"
                        }
                    },
                },
                "required": [
                    "chapters",
                    "summary",
                    "key_questions",
                    "notepad",
                ],
            },
            "aggregation": {
                "type": "object",
                "properties": {
                    "key_issues_discussed": {
                        "type": "array",
                        "description": "List of sub-topics discussed in the conversation based on the call transcript. Try to keep the sub-topic titles brief.",
                        "items": {"type": "string"},
                    },
                    "action_items": {
                        "type": "array",
                        "description": "List of tasks to be done by parties involved in the conversation based on the call transcript.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "assignee": {
                                    "type": "string",
                                    "description": "Role of the person in the transcript responsible for performing the action item."
                                },
                                "action_item": {
                                    "type": "string",
                                    "description": "Action to be done by the respective assignee."
                                },
                                "deadline": {
                                    "type": "string",
                                    "description": "Deadline for the action to be done by the assignee."
                                }
                            }
                        },
                    },
                },
                "required": ["key_issues_discussed", "action_items"],
            },
            "intent_sources":{
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "keyword": {
                            "type": "string"
                        },
                        "sources": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties":{
                                    "speaker": {
                                        "type": "string"
                                    },
                                    "source": {
                                        "type": "string",
                                        "description": "A list of sentences or phrases (sources) from the transcript where the keyword or its corresponding translation in other languages or semantic references appear.",
                                    }
                                }
                                
                            }
                        },
                    },
                    "required": ["keyword", "sources"],
                },
            },
        },
        "required":[
            "summary",
            "aggregation",
            "intent_sources"
        ]
    }

combined_components_2_response_schema = {
        "type": "object",
        "properties":{
            "entity_filler_count":{
                "type": "object",
                "properties": {
                    "entities": {
                        "type": "array",
                        "description": "List of entities proper nouns mentioned in the transcript",
                        "items": {
                            "type": "string",
                            "description": "name of the entity",
                        },
                    },
                    "filler_words": {
                        "type": "array",
                        "description": "List of fillers used in the conversation according to the transcript.",
                        "items":{
                            "type": "string",
                            "description": "filler word present in the transcript"
                        }
                    },
                },
                "required": ["entities", "filler_words"],
            },
            "customer_insights":{
                "type": "object",
                "properties": {
                    "customer_insights": {
                        "type": "object",
                        "properties": {
                            "personal_information": {
                                "type": "object",
                                "properties": {
                                    "name": {
                                        "type": "object",
                                        "properties": {
                                            "first_name": {"type": "string"},
                                            "last_name": {"type": "string"},
                                        },
                                    },
                                    "contact": {
                                        "type": "object",
                                        "properties": {
                                            "phone_number": {"type": "string"},
                                            "email": {"type": "string"},
                                        },
                                    },
                                    "address": {
                                        "type": "object",
                                        "properties": {
                                            "city": {"type": "string"},
                                            "state": {"type": "string"},
                                            "postal_code": {"type": "string"},
                                            "country": {"type": "string"},
                                        },
                                    },
                                    "company": {
                                        "type": "object",
                                        "properties": {
                                            "company_name": {"type": "string"},
                                        },
                                    },
                                },
                                "required": ["name", "contact", "address"],
                            },
                            "customer_issue": {
                                "type": "object",
                                "properties": {
                                    "primary_concern": {"type": "string"},
                                    "issue_details": {"type": "string"},
                                    "urgency_level": {
                                        "type": "string",
                                        "enum": ["low", "medium", "high"],
                                    },
                                },
                                "required": [
                                    "primary_concern",
                                    "issue_details",
                                    "urgency_level",
                                ],
                            },
                            "customer_feedback": {
                                "type": "object",
                                "properties": {
                                    "explicit_feedback": {"type": "string"},
                                    "satisfaction_level": {
                                        "type": "string",
                                        "enum": ["satisfied", "neutral", "dissatisfied"],
                                    },
                                },
                                "required": ["explicit_feedback", "satisfaction_level"],
                            },
                        },
                        "required": [
                            "personal_information",
                            "customer_issue",
                            "customer_feedback",
                        ],
                    }
                },
                "required": ["customer_insights"],
            }
        },
        "required":[
            "entity_filler_count",
            "boolean_checks",
            "customer_insights"
        ]
    }