package cz.zcu.kgm.didymos;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.apache.commons.math3.linear.*;
import org.apache.commons.io.IOUtils;
import java.io.IOException;
import java.io.InputStream;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;


public class CoordinateConverter {
    public static final int DIMENSION = 2;
    public static final int HOMOGENEOUS_DIMENSION = DIMENSION + 1;

    public static class Rect {
        @JsonProperty("x_left") double xLeft;
        @JsonProperty("y_top") double yTop;
        @JsonProperty("width") double width;
        @JsonProperty("height") double height;
    }

    public static class MapPosition {
        @JsonProperty("x") double x;
        @JsonProperty("y") double y;
    }

    //@JsonIgnoreProperties
    public static class TrackedObject {
        @JsonIgnore int frameNumber = 0;
        @JsonProperty("class_name") String className;
        @JsonProperty("class_id") int classId;
        @JsonProperty("image_rect") Rect imageRect;
        @JsonProperty("map_position") MapPosition mapPosition;
        @JsonProperty("track_id") int trackId;
    }

    public static void main(String[] args) {
        if (args.length < 1) {
            System.err.println("Please provide the filename as command line argument.");
            return;
        }
        String filename = args[0];

        Map<RealVector, RealVector> transformationPoints = readTransformationPoints(filename);
        if (transformationPoints == null)
            return;
        RealMatrix transformation = estimateTransformation(transformationPoints);
        List<TrackedObject> objects = readTrackedObjects(System.in);
/*
List<TrackedObject> tst = objects.subList(0, transformationPoints.size());
int i = 0;
for(RealVector pos : transformationPoints.keySet()) {
    TrackedObject o = tst.get(i);
    o.mapPosition.x = pos.getEntry(0) / pos.getEntry(HOMOGENEOUS_DIMENSION - 1);
    o.mapPosition.y = pos.getEntry(1) / pos.getEntry(HOMOGENEOUS_DIMENSION - 1);
    i++;
}
objects = tst;
*/
        writeObjects(objects, transformation, System.out);
    }

    /**
     * Read JSON from InputStream and returns parsed point coordinates in a matrix (one point per column).
     * @param stream    stream with point coordinates in JSON
     * @return  matrix with point coordinates
     */
    private static List<TrackedObject> readTrackedObjects(InputStream stream) {
        ObjectMapper objectMapper = new ObjectMapper();

        // Read coordinates from file
        JsonNode frames;
        try {
            String input = IOUtils.toString(stream, StandardCharsets.UTF_8);
            frames = objectMapper.readTree(input);
        } catch (IOException e) {
            System.err.println("Error reading input data.");
            return null;
        }

        // Extract tracking features
        if (!frames.isArray()) {
            System.err.println("Error reading tracking features - arrays of frames not found.");
            return null;
        }

        List<TrackedObject> result = new ArrayList<>(100);
        for (JsonNode f : frames) {
            JsonNode node;
            node = f.get("frame_number");
            int frameNumber = node.asInt(0);
            node = f.get("objects");
            if (!node.isArray()) {
                System.err.println("Error reading tracking features array in frame " + frameNumber + ".");
                continue;
            }
            ArrayNode objectArray = (ArrayNode) node;
            for (JsonNode o : objectArray) {
                if (!o.isObject()) {
                    System.err.println("Error reading tracking feature in frame " + frameNumber + ".");
                    continue;
                }
                try {
                    TrackedObject pos = objectMapper.treeToValue(o, TrackedObject.class);
                    pos.frameNumber = frameNumber;
                    result.add(pos);
                } catch (JsonProcessingException e) {
                    System.err.println("Error reading tracking feature in frame " + frameNumber + ".");
                    return null;
                }
            }
        }
        return result;
    }

    /**
     * Transforms point coordinates with transformation matrix transformation.
     * @param points            point coordinates to transform
     * @param transformation    transformation matrix
     * @return  transformed point coordinates
     */
/*
    private static RealMatrix transformCoordinates(RealMatrix points, RealMatrix transformation) {
        return transformation.multiply(points);
    }
*/

    /**
     * Writes points to PrintStream as GeoJSON.
     * @param objects   list of tracked objects
     * @param transformation   coordinate transformation matrix
     * @param stream    stream to write the GeoJSON to
     */
    private static void writeObjects(List<TrackedObject> objects, RealMatrix transformation, PrintStream stream) {
        ObjectMapper objectMapper = new ObjectMapper();
        objectMapper.enable(SerializationFeature.INDENT_OUTPUT);
        ObjectNode geoJson = objectMapper.createObjectNode();
        geoJson.put("type", "FeatureCollection");

        //geoJson.put("name", "points");
        ArrayNode features = objectMapper.createArrayNode();
        geoJson.set("features", features);
        for (TrackedObject o : objects) {
            // Create geojson feature
            ObjectNode feature = objectMapper.createObjectNode();
            feature.put("type", "Feature");

            // properties
            ObjectNode properties = objectMapper.createObjectNode();
            properties.put("class_name", o.className);
            properties.put("class_id", o.classId);
            properties.put("track_id", o.trackId);
            properties.put("frame_number", o.frameNumber);
            properties.put("map_position.x", o.mapPosition.x);
            properties.put("map_position.y", o.mapPosition.y);
            properties.put("image_rect.x_left", o.imageRect.xLeft);
            properties.put("image_rect.y_top", o.imageRect.yTop);
            properties.put("image_rect.width", o.imageRect.width);
            properties.put("image_rect.height", o.imageRect.height);
            feature.set("properties", properties);

            // geometry
            //double[] pos = new double[]{o.mapPosition.x, o.mapPosition.y, 1.0};
            double[] pos = new double[]{o.imageRect.xLeft + o.imageRect.width / 2, o.imageRect.yTop + o.imageRect.height, 1.0};
            double[] transformedPos = transformation.operate(pos);
            ObjectNode geometry = objectMapper.createObjectNode();
            geometry.put("type", "Point");
            geometry.putPOJO("coordinates",
                    new double[]{transformedPos[0] / transformedPos[HOMOGENEOUS_DIMENSION - 1], transformedPos[1] / transformedPos[HOMOGENEOUS_DIMENSION - 1]});
            feature.set("geometry", geometry);
            features.add(feature);
        }

        // Write geoJSON to stream
        try {
            stream.println(objectMapper.writeValueAsString(geoJson));
        } catch (IOException e) {
            System.err.println("Error writing transformed coordinates: " + e.getMessage());
        }
    }

    /**
     * Estimates homography between two sets of coordinates of transformation points.
     * @param points    corresponding coordinates of transformation points as key-value pairs
     * @return  homography matrix
     */
    private static RealMatrix estimateTransformation(Map<RealVector, RealVector> points) {
        int minPointCount = (HOMOGENEOUS_DIMENSION * HOMOGENEOUS_DIMENSION - 1) / 2;
            // Transformation matrix has HOMOGENEOUS_DIMENSIONS^2 elements. It is defined up to scale,
            // so total number of degrees of freedom is HOMOGENEOUS_DIMENSIONS^2 - 1.
            // Each point correspondence gives 2 degrees of freedom.
        if (points.size() < minPointCount)
            throw new IllegalArgumentException("The number of point correspondences is too low. The minimum is "
                    + minPointCount + ".");
        RealMatrix A = new Array2DRowRealMatrix(2 * points.size(), 3 * HOMOGENEOUS_DIMENSION);
        int rowCounter = 0;
        for(Map.Entry<RealVector, RealVector> entry : points.entrySet()) {
            RealVector inCoords = entry.getKey();
            RealVector outCoords = entry.getValue();
            for(int i = 0; i < HOMOGENEOUS_DIMENSION; i++) {
                //A.setEntry(rowCounter, HOMOGENEOUS_DIMENSION + i,
                //        -outCoords.getEntry(2) * inCoords.getEntry(i));
                //A.setEntry(rowCounter, 2 * HOMOGENEOUS_DIMENSION + i,
                //        outCoords.getEntry(1) * inCoords.getEntry(i));
                //A.setEntry(rowCounter + 1, i,
                //        outCoords.getEntry(2) * inCoords.getEntry(i));
                //A.setEntry(rowCounter + 1, 2 * HOMOGENEOUS_DIMENSION + i,
                //        -outCoords.getEntry(0) * inCoords.getEntry(i));
                A.setEntry(rowCounter, HOMOGENEOUS_DIMENSION + i,
                        inCoords.getEntry(i));
                A.setEntry(rowCounter, 2 * HOMOGENEOUS_DIMENSION + i,
                        -outCoords.getEntry(1) * inCoords.getEntry(i));
                A.setEntry(rowCounter + 1, i,
                        inCoords.getEntry(i));
                A.setEntry(rowCounter + 1, 2 * HOMOGENEOUS_DIMENSION + i,
                        -outCoords.getEntry(0) * inCoords.getEntry(i));
            }
            rowCounter += 2;
        }
        RealVector solution = null;
        /*
        if (points.size() == minPointCount) { // exact solution
            DecompositionSolver solver = new LUDecomposition(A).getSolver();
            RealVector constants = new ArrayRealVector(A.getRowDimension());
            RealVector solution = solver.solve(constants);
            solution.mapMultiplyToSelf(1.0 / solution.getNorm());
        } else { // over-determined solution
         */ {
            SingularValueDecomposition decomposition = new SingularValueDecomposition(A);
            double[] eigenNumber = decomposition.getSingularValues();
            /*
            int minIdx = 0;
            double minEigenNumber = eigenNumber[0];
            for (int i = 1; i < eigenNumber.length; i++)
                if (minEigenNumber > eigenNumber[i]) {
                    minEigenNumber = eigenNumber[i];
                    minIdx = i;
                }
            RealMatrix V = decomposition.getV();
            solution = decomposition.getV().getColumnVector(minIdx);
             */
            solution = decomposition.getV().getColumnVector(3 * HOMOGENEOUS_DIMENSION - 1);
        }
        RealMatrix transformation = new Array2DRowRealMatrix(HOMOGENEOUS_DIMENSION, HOMOGENEOUS_DIMENSION);
        for(int i = 0; i < solution.getDimension(); i++)
            transformation.setEntry(i / HOMOGENEOUS_DIMENSION, i % HOMOGENEOUS_DIMENSION, solution.getEntry(i));
        return transformation;
    }


    /**
     * Creates new vector and sets its fields to values extracted from node.
     * @param node   ArrayNode containing the coordinates to extract
     * @return  vector with fields filled with coordinates (in homogeneous representation)
     */
    private static RealVector extractCoordinates(ArrayNode node) {
        double[] coords = new double[HOMOGENEOUS_DIMENSION];
        try {
            for (int i = 0; i < DIMENSION; i++)
                coords[i] = node.has(i) ? node.get(i).asDouble() : 0;
            coords[HOMOGENEOUS_DIMENSION - 1] = 1;
        } catch (IllegalArgumentException e) {
            System.err.println("Error transforming coordinates: " + e.getMessage());
            return null;
        }
        return new ArrayRealVector(coords);
    }

    /**
     * Reads point coordinates in input and output coordinate systems from json in filename.
     * The input coordinates can be either 2D or 3D. If the coordinates are only 2D, then value of 0
     * is assumed as the third coordinate.
     * @param filename  the file from that the coordinates are read
     * @return  2xnx3 array with input [0][][] and output [1][][] 3D coordinates
     */
    private static Map<RealVector, RealVector> readTransformationPoints (String filename) {
        ObjectMapper objectMapper = new ObjectMapper();

        // Read coordinate tuples from file
        JsonNode coordinateTuples;
        try {
            String fileContent = new String(Files.readAllBytes(Paths.get(filename)));
            coordinateTuples = objectMapper.readTree(fileContent);
        } catch (IOException e) {
            System.err.println("Error reading file: " + e.getMessage());
            return null;
        }

        // Extract coordinate tuples
        Map<RealVector, RealVector> points = new LinkedHashMap<>();
        int counter = 0;
        for (JsonNode tuple : coordinateTuples) {
            counter++;
            if (!tuple.isArray()) {
                System.err.printf("Error reading coordinate pair no. %d (%s)", counter, tuple.textValue());
                continue;
            }
            JsonNode inCoordsNode = tuple.get(0);
            JsonNode outCoordsNode = tuple.get(1);
            if (!inCoordsNode.isArray() || !outCoordsNode.isArray()) {
                System.err.printf("Error reading coordinate pair no. %d (%s)", counter, tuple.textValue());
                continue;
            }
            RealVector inCoords = extractCoordinates((ArrayNode) inCoordsNode);
            RealVector outCoords = extractCoordinates((ArrayNode) outCoordsNode);
            if (inCoords == null || outCoords == null) {
                System.err.printf("Error reading coordinate pair no. %d (%s)", counter, tuple.textValue());
                continue;
            }
            points.put(inCoords, outCoords);
        }
        return points;
    }




}
