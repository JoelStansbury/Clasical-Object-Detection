# Classical-Object-Detection
Object detection for computer vision using SKlearn's KMeans and DBSCAN

### Disclaimer
There is very likely someone who has implemented this much better than I have, but I like to learn so that is the reason for making this.

## Technical Details

### The basic idea is quite simple. 
_If there is a group of pixels (close in proximity) within an image that are similar in color, then they probably correspond to an object._

### How do we find similar colors?

You guessed it! __KMeans!__ First we unravel the image into an __N x 3__ dimentional matrix, where N is the number of pixels in the image. Next, we simply pass that feature matrix into a KMeans model looking for some predefined number of primary colors. This predefined number (`Q`) is one of the parameters of the function.

### How do we find clusters?

Now for each of the `Q` primary colors we found using KMeans we need to find clusters of pixels close to the respective color. This is done using DBSCAN. DBSCAN requires us to specify the maximum distance between members of the same cluster, this is what `eps` does. In the context of pixels an `eps` of __1.0__ means all members of the same cluster must be immediately adjacent to another member in order to be considered a member.

I use Euclidean distance (p=2).

## Usage
