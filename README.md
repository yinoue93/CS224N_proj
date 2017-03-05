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
| Song Type       | Song Genre                                                                   |     R    |     17     | Reel, Jig, Hornpipe |
| Time Signature  | Specifies how many beats are in each bar and which note value gets one beat  |     M    |     15     | 4/4, 6/8, 3/4       |
| Note Unit Size  | Specifies which note value gets one beat in the text file                    |     L    |      3     | 1/8, 1/4, 1/16      |
| Number of Flats | Positive for songs with flats, 0 for neutral, negative for songs with sharps |     K    |     12     | -1, -2, -3          |
| Song Mode       | 0=Major, 1=Minor, 2=Mixolydian, 3=Dorian, 4=Phrygian, 5=Lydian, 6=Locrian    |     K    |      6     | 0, 1, 3             |
| Song Length     | Number of measures in a song                                                 |          |            |                     |
| Song Complexity | Busy-ness of a song.                                                         |          |            |                     |

The song portion of the numpy array is **119** dimensions.
