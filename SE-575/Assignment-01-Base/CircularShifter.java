import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class CircularShifter {

	public ArrayList<String> shiftLines(List<String> lines) {
		ArrayList<String> allShiftedLines = new ArrayList<String>();
		for (String line : lines)
			allShiftedLines.addAll(getLineShifts(line));
		return allShiftedLines;
	}

	private static ArrayList<String> getLineShifts(String line) {
		ArrayList<String> allShifts = new ArrayList<String>();
		ArrayList<String> currentTokens = tokenizeLine(line);
		for (int i = 0; i < currentTokens.size(); i++) {
			currentTokens = shiftElementsOnce(currentTokens);
			allShifts.add(String.join(" ", currentTokens));
		}
		return allShifts;
	}

	private static ArrayList<String> shiftElementsOnce(List<String> list) {
		ArrayList<String> temp = new ArrayList<String>();
		temp.add(list.get(list.size() - 1)); // add last element
		temp.addAll(list.subList(0, list.size() - 1)); // add all but the last element
		return temp;
	}

	private static ArrayList<String> tokenizeLine(String line) {
		return new ArrayList<String>(Arrays.asList(line.split(" ")));
	}

}
