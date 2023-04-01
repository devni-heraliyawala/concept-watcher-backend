import re
import html

CONTRACTION_MAP = {
"ain't": "is not",
"aren't": "are not",
"cannot": "can not",
"can't": "can not",
"can't've": "cannot have",
"cause":"because",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"daren't":"dare not",
"dasn't":"dare not",  
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"dont":"do not",
"everyone's":"everyone", # This is not a contraction. It's a possession
"finna": "fixing to",
"gimme":"give me",
"gonna": "going to",
"gotta":"got to",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he would",
"he'd've": "he would have",
"he'll": "he will",
"he'll've": "he he will have",
"he's": "he is",
"here's":"here is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how is",
"I'd": "I would",
"I'd've": "I would have",
"I'll": "I will",
"I'll've": "I will have",
"I'm": "I am",
"I've": "I have",
"i'd": "i would",
"i'd've": "i would have",
"i'll": "i will",
"i'll've": "i will have",
"i'm": "i am",
"i've": "i have",
"isn't": "is not",
"isnt":"is not",
"it'd": "it would",
"it'd've": "it would have",
"it'll": "it will",
"it'll've": "it will have",
"it's": "it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "she would",
"she'd've": "she would have",
"she'll": "she will",
"she'll've": "she will have",
"she's": "she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so as",
"that'd": "that would",
"that'd've": "that would have",
"that's": "that is",
"there'd": "there would",
"there'd've": "there would have",
"there's": "there is",
"they'd": "they would",
"they'd've": "they would have",
"they'll": "they will",
"they'll've": "they will have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
"we'd": "we would",
"we'd've": "we would have",
"we'll": "we will",
"we'll've": "we will have",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what will",
"what'll've": "what will have",
"what're": "what are",
"what's": "what is",
"what've": "what have",
"when's": "when is",
"when've": "when have",
"where'd": "where did",
"where's": "where is",
"where've": "where have",
"who'll": "who will",
"who'll've": "who will have",
"who's": "who is",
"who've": "who have",
"why's": "why is",
"why've": "why have",
"will've": "will have",
"won't": "will not",
"won't've": "will not have",
"would've": "would have",
"wouldn't": "would not",
"wouldn't've": "would not have",
"y'all": "you all",
"y'all'd": "you all would",
"y'all'd've": "you all would have",
"y'all're": "you all are",
"y'all've": "you all have",
"you'd": "you would",
"you'd've": "you would have",
"you'll": "you will",
"you'll've": "you will have",
"you're": "you are",
"you've": "you have",
"youre": "you are"
}

def preprocess_text(text):
    # Apply the preprocessing functions to the text column
    text =remove_html_tag(text)
    text =lowercase(text)
    text =apply_contraction_map(text)
    text =preprocess(text)
    text =remove_escape_sequences(text)
    text =remove_ansi_escape_sequences(text)
    text =remove_dash_underscore(text)
    text =remove_non_alphanumeric(text)
    text =remove_whitespace_between_words(text)
    text =remove_whitespace_between_sentences(text)
    return text

def expand_contractions(sentence, contraction_mapping):
    
    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())), 
                                      flags=re.IGNORECASE|re.DOTALL)
    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match)\
                                if contraction_mapping.get(match)\
                                else contraction_mapping.get(match.lower())                       
        expanded_contraction = first_char+expanded_contraction[1:]
        return expanded_contraction
        
    expanded_sentence = contractions_pattern.sub(expand_match, sentence)
    return expanded_sentence

def remove_html_tag(text):
    text=html.unescape(text)
    return re.sub(r'<.*?>',' ',text)

def lowercase(text):
    return text.lower()

def apply_contraction_map(text):
  return expand_contractions(text,CONTRACTION_MAP)

def preprocess(x):
    x = x.replace(",000,000", " m").replace(",000", " k").replace("′", "'").replace("’", "'")\
                           .replace("%", " percent ").replace("₹", " rupee ").replace("$", " dollar ")\
                           .replace("€", " euro ")
    
    x = re.sub(r"([0-9]+)000000", r"\1m", x)
    x = re.sub(r"([0-9]+)000", r"\1k", x)
    x=re.sub(r'((www\.[^\s]+)|(https?://[^\s]+))',' ',x)
    x=re.sub(r"\\s*\\b(?=\\w*(\\w)\\1{2,})\\w*\\b",' ',x)
    x=re.sub('[^a-zA-Z]',' ',x)
    x=''.join([i for i in x if not i.isdigit()])
    return x

def remove_escape_sequences(text):
  return re.sub(r'[\t\n\r\f\v]','',text)
  
def remove_ansi_escape_sequences(text):
  return re.sub(r'\x1b[^m]*m','',text)

def remove_dash_underscore(text):
  return re.sub(r'[-_.]',' ',text)

def remove_non_alphanumeric(text):
  return re.sub(r'^a-zA-Z0-9 ',' ',text)

def remove_whitespace_between_words(text):
  return re.sub(r'\s+',' ',text)

def remove_whitespace_between_sentences(text):
  return re.sub(r'^\s+|\s+$','',text)
