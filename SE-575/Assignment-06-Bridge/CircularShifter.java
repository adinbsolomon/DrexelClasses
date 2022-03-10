import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class CircularShifter extends Filter {

	public CircularShifter(Pipe inPipe, Pipe outPipe) {
		super(inPipe, outPipe);
	}

	@Override
	public void filter() {
		while (this.inPipe.isNotEmptyOrIsNotClosed()) {
			this.shiftLine(this.inPipe.read());
		}
		this.outPipe.close();
		this.stop();
	}

	private void sendShiftedLineToOutPipe(String line) {
		this.outPipe.write(line);
	}

	private void shiftLine(String line) {
		ArrayList<String> currentTokens = tokenizeLine(line);
		for (int i = 0; i < currentTokens.size(); i++) {
			currentTokens = shiftElementsOnce(currentTokens);
			this.sendShiftedLineToOutPipe(String.join(" ", currentTokens));
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
