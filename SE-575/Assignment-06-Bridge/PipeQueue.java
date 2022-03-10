import java.util.Queue;
import java.util.concurrent.ConcurrentLinkedQueue;

public class PipeQueue extends Pipe {

	protected Queue<String> queue;

	public PipeQueue() {
		super();
		this.queue = new ConcurrentLinkedQueue<>();
	}

	@Override
	public void write(String string) {
		this.queue.offer(string);
	}

	@Override
	public String read() {
		return this.queue.poll();
	}

	@Override
	public boolean isEmpty() {
		return this.queue.isEmpty();
	}

}
