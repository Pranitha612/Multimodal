class SentenceGenerator:
    def generate(self, emotions, relationships):
        """Generate natural language description"""
        if not emotions:
            return "No people detected in the image."
            
        num_people = len(emotions)
        emotion_counts = {}
        
        for emotion_data in emotions:
            emotion = emotion_data['emotion']
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            
        dominant_emotion = max(emotion_counts, key=emotion_counts.get)
        
        relationship_type = 'default'
        if relationships:
            rel_counts = {}
            for rel in relationships:
                rel_type = rel['relationship']
                rel_counts[rel_type] = rel_counts.get(rel_type, 0) + 1
            dominant_relation = max(rel_counts, key=rel_counts.get)
        else:
            dominant_relation = 'strangers'
            
        sentence_parts = []
        
        if num_people == 2:
            sentence_parts.append("Two people")
        else:
            sentence_parts.append(f"{num_people} people")
            
        if dominant_relation != 'strangers':
            sentence_parts.append(f"who appear to be {dominant_relation}")
            
        if len(set([e['emotion'] for e in emotions])) == 1:
            sentence_parts.append(f"all looking {dominant_emotion}")
        else:
            emotion_list = [f"{count} {emotion}" for emotion, count in emotion_counts.items()]
            sentence_parts.append(f"with {', '.join(emotion_list)} expressions")
            
        description = " ".join(sentence_parts) + "."
        
        details = []
        for i, emotion_data in enumerate(emotions, 1):
            emotion = emotion_data['emotion']
            confidence = emotion_data['confidence']
            if confidence > 0.7:
                details.append(f"Person {i} appears {emotion}")
                
        if details and len(details) <= 3:
            description += " " + ". ".join(details) + "."
            
        return description

def generate_advanced_description(emotions, relationships, use_transformer=False):
    generator = SentenceGenerator()
    return generator.generate(emotions, relationships)