public abstract class Filter implements Runnable {

	Pipe inPipe;
	Pipe outPipe;
	private boolean running = false;

	public Filter(Pipe inPipe, Pipe outPipe) {
		this.inPipe = inPipe;
		this.outPipe = outPipe;
	}

	abstract void filter();

	public void start() {
		new Thread(this).start();
	}

	public void stop() {
		running = false;
	}

	@Override
	public void run() {
		running = true;
		while (running) {
			filter();
		}
	}
}
