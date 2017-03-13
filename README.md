# CS224N Final Project

## Files
  * `midi_crawler.py` - crawls the Internet for .mid files. 
    * Flags: **-u**: url, **-f**: output folder name, **-d**: crawl depth, **-r**: crawl regEx rules
  * `utils_preprocess.py` - utility script for midi preprocessing.

## Useful Websites
<http://www.mandolintab.net/abcconverter.php>

<http://abcnotation.com/wiki/abc:standard:v2.1>

## Example .abc format

![exABC](images/example_abc.png?raw=true "Example .abc Music")

```
X:1
T:NeilyCleere's
R:polka
M:2/4
L:1/8
K:Dmaj
Q:100
FG|A2A>B|=c/B/AFG|AB=c/B/A|G2FG|A>^GA>B|=c/B/Af2|edAF|G2:||:fg|a>gfa|gefg|a>gfa|g2fg|a>gfa|gef2|edAF|G2:|
```

## Data Encoding Structure
The numpy array representing each sample is composed of two parts: the metadata and the song.

The first **7** integers in the numpy array are the metadata. They are, in order: **song type (R)**, **time signature (M)**, **note unit size (L)**, **number of flats (K)**, **song mode (K)**, **length**, **complexity**.

Length is calculated by counting the distinct number of times the character '|' appears in a file, and complexity is calculated by (*number of notes in a song*) x 100/(*len* x *number of beats in a measure*). In other words, the complexity measure is trying to estimate how *busy* a song is.

|                 | Description                                                                  | .abc Tag | Dimensions | Examples (Top 3)    |
|-----------------|------------------------------------------------------------------------------|----------|------------|---------------------|
| Song Type       | Song Genre                                                                   |     R    |     16     | Reel, Jig, Hornpipe |
| Time Signature  | Specifies how many beats are in each bar and which note value gets one beat  |     M    |     15     | 4/4, 6/8, 3/4       |
| Note Unit Size  | Specifies which note value gets one beat in the text file                    |     L    |      3     | 1/8, 1/4, 1/16      |
| Number of Flats | Positive for songs with flats, 0 for neutral, negative for songs with sharps |     K    |     12     | -1, -2, -3          |
| Song Mode       | 0=Major, 1=Minor, 2=Mixolydian, 3=Dorian, 4=Phrygian, 5=Lydian, 6=Locrian    |     K    |      6     | 0, 1, 3             |
| Song Length     | Number of measures in a song                                                 |          |            |                     |
| Song Complexity | Busy-ness of a song.                                                         |          |            |                     |

The song portion of the numpy array is **82** dimensions (i.e. **80** music characters and **2** BEGIN/END special characters).

## Metadata and Music Encoding Map
```
>>> pickle.load(open('vocab_map_meta.p'))
{'R': {'jig': 0, 'waltz': 1, 'three-two': 2, 'songair': 3, 'slowair': 4, 'strathspey': 5, 
	'polka': 6, 'air': 7, 'barndance': 8, 'slide': 9, 'slipjig': 10, 'hornpipe': 11, 
	'mazurka': 12, 'reel': 13, 'highlandfling': 14, 'quickstep': 15}, 
'M': {'7/8': 1, '11/8': 2, '5/4': 0, '6/8': 3, '5/8': 4, '4/4': 5, '6/4': 6, '13/8': 7, 
	'3/2': 8, '3/4': 9, '9/8': 10, '12/8': 11, '2/2': 12, '9/4': 13, '2/4': 14}, 
'L': {'1/4': 0, '1/16': 1, '1/8': 2}, 
'K_key': {'-5': 0, '-4': 1, '1': 2, '0': 3, '3': 4, '-6': 5, '-1': 6, '4': 7, '-3': 8, 
	'-2': 9, '2': 10, '5': 11}, 
'K_mode': {'1': 0, '0': 1, '3': 2, '2': 3, '5': 4, '4': 5}}
```

```
>>> pickle.load(open('vocab_map_music.p'))
{'!': 0, ' ': 1, '#': 2, "'": 3, '&': 4, ')': 5, '(': 6, '+': 7, '*': 8, '-': 9, ',': 10, 
'/': 11, '.': 12, '1': 13, '0': 14, '3': 15, '2': 16, '5': 17, '4': 18, '7': 19, '6': 20, 
'9': 21, '8': 22, ':': 23, '=': 24, '<': 25, '>': 26, 'A': 27, 'C': 28, 'B': 29, 'E': 30, 
'D': 31, 'G': 32, 'F': 33, 'H': 34, 'K': 35, 'J': 36, 'M': 37, 'L': 38, 'O': 39, 'Q': 40, 
'P': 41, 'S': 42, 'R': 43, 'U': 44, 'T': 45, 'V': 46, '[': 47, ']': 48, '\\': 49, '_': 50, 
'^': 51, 'a': 52, 'c': 53, 'b': 54, 'e': 55, 'd': 56, 'g': 57, 'f': 58, 'i': 59, 'h': 60, 
'j': 61, 'm': 62, 'l': 63, 'o': 64, 'n': 65, 'p': 66, 's': 67, 'r': 68, 'u': 69, 't': 70, 
'w': 71, 'v': 72, 'y': 73, 'x': 74, '{': 75, 'z': 76, '}': 77, '|': 78, '~': 79}
```

## Sherlock access
* Ger a kerberos ticket using kinit nipuna1@stanford.edu
* Log into nipuna1@sherlock.stanford.edu
* Run the command 'srun -p gpu --qos gpu --gres gpu:1 --pty bash' and remember the gpu node you are assigned
* Treat the machine like a normal GPU interactive session i.e. run tests, screens etc
* To review a long job over screen, log into Sherlock and then type the command 'ssh gpu-[0-9]-[0-9]' depending on the node you were allocated the first time you ran the job. You will be able to log into the node and then proceed as normal
