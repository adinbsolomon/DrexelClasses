import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class CircularShifter extends Colleague {

	public CircularShifter(Mediator mediator) {
		super(mediator);
	}

	public void shiftLine(String line) {
		ArrayList<String> currentTokens = tokenizeLine(line);
		for (int i = 0; i < currentTokens.size(); i++) {
			currentTokens = shiftElementsOnce(currentTokens);
			this.mediator.shiftedLine(String.join(" ", currentTokens));
		}
	}

	private ArrayList<String> shiftElementsOnce(List<String> list) {
		ArrayList<String> temp = new ArrayList<String>();
		temp.add(list.get(list.size() - 1)); // add last element
		temp.addAll(list.subList(0, list.size() - 1)); // add all but the last element
		return temp;
	}

	private ArrayList<String> tokenizeLine(String line) {
		return new ArrayList<String>(Arrays.asList(line.split(" ")));
	}

}
