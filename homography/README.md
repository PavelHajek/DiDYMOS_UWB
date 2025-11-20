# Conversion of detected features

This tool is used to convert a JSON file with detected features into a GeoJSON file. The conversion includes the transformation of image coordinates into geographic coordinates according to a transformation calculated on the basis of at least 4 pairs of corresponding image and geographic coordinates.

The first parameter of the command line program specifies the name of the file containing corresponding pairs of image and geographic coordinates in JSON format. An example of such file is `gcp.json`.

The tool computes the transformation and then reads detected features in JSON format from standard input, processes them and writes them to standard output as GeoJSON. An example of the input and output files is `tracked_objects.json` and `tracked_objects_wgs84.geojson`.

Usage
```
    java -jar path/to/perspective2map.jar gcp.json <tracked_objects.json >tracked_objects.json
```

A source code of this tool is in `src` directory. A compiled Java JAR file in `out` directory.
