public class MediatorMasterControl implements Mediator {

	private Input input;
	private CircularShifter shifter;
	private Alphabetizer sorter;
	private Output output;

	public MediatorMasterControl() {
		this.input = new Input(this);
		this.shifter = new CircularShifter(this);
		this.sorter = new Alphabetizer();
		this.output = new Output();
	}

	public void start() {
		this.input.read();
	}

	@Override
	public void inputHasNextLine(String line) {
		this.shifter.shiftLine(line);
	}

	@Override
	public void inputIsFinished() {
		this.output.write(this.sorter.getAlphabetizedLines());
	}

	@Override
	public void shiftedLine(String line) {
		this.sorter.addToAlphabetizedSet(line);
	}

}
