package GridworldDomain;


import java.awt.Color;
import java.util.List;
import java.util.ArrayList;
import java.util.Arrays;

import Amdp.AMDPAgent;
import Amdp.AMDPPolicyGenerator;
import burlap.oomdp.auxiliary.common.GoalConditionTF;
import burlap.behavior.singleagent.auxiliary.EpisodeSequenceVisualizer;
import burlap.behavior.singleagent.planning.deterministic.DDPlannerPolicy;
import burlap.behavior.singleagent.planning.deterministic.uninformed.bfs.BFS;
import burlap.behavior.singleagent.planning.stochastic.valueiteration.ValueIteration;
import burlap.behavior.policy.Policy;
import burlap.behavior.singleagent.EpisodeAnalysis;
import burlap.behavior.singleagent.auxiliary.StateReachability;
import burlap.behavior.singleagent.auxiliary.performance.LearningAlgorithmExperimenter;
import burlap.behavior.singleagent.auxiliary.performance.PerformanceMetric;
import burlap.behavior.singleagent.auxiliary.performance.TrialMode;
import burlap.behavior.singleagent.auxiliary.valuefunctionvis.ValueFunctionVisualizerGUI;
import burlap.behavior.singleagent.auxiliary.valuefunctionvis.common.ArrowActionGlyph;
import burlap.behavior.singleagent.auxiliary.valuefunctionvis.common.LandmarkColorBlendInterpolation;
import burlap.behavior.singleagent.auxiliary.valuefunctionvis.common.PolicyGlyphPainter2D;
import burlap.behavior.singleagent.auxiliary.valuefunctionvis.common.StateValuePainter2D;
import burlap.behavior.singleagent.learning.LearningAgent;
import burlap.behavior.singleagent.learning.LearningAgentFactory;
import burlap.behavior.singleagent.learning.tdmethods.QLearning;
import burlap.behavior.singleagent.learning.tdmethods.SarsaLam;
import burlap.behavior.singleagent.planning.Planner;
import burlap.behavior.singleagent.planning.deterministic.DeterministicPlanner;
import burlap.behavior.singleagent.planning.deterministic.informed.Heuristic;
import burlap.behavior.singleagent.planning.deterministic.informed.astar.AStar;
import burlap.behavior.singleagent.planning.deterministic.uninformed.dfs.DFS;
import burlap.behavior.valuefunction.ValueFunction;
import burlap.domain.singleagent.gridworld.GridWorldDomain;
import burlap.domain.singleagent.gridworld.GridWorldVisualizer;
import burlap.oomdp.auxiliary.common.SinglePFTF;
import burlap.oomdp.auxiliary.stateconditiontest.StateConditionTest;
import burlap.oomdp.auxiliary.stateconditiontest.TFGoalCondition;
import burlap.oomdp.core.Domain;
import burlap.oomdp.core.GroundedProp;
import burlap.oomdp.core.PropositionalFunction;
import burlap.oomdp.core.TerminalFunction;
import burlap.oomdp.core.objects.ObjectInstance;
import burlap.oomdp.core.states.State;
import burlap.oomdp.singleagent.RewardFunction;
import burlap.oomdp.singleagent.SADomain;
import burlap.oomdp.singleagent.common.GoalBasedRF;
import burlap.oomdp.singleagent.common.UniformCostRF;
import burlap.oomdp.singleagent.environment.Environment;
import burlap.oomdp.singleagent.environment.SimulatedEnvironment;
import burlap.oomdp.statehashing.HashableStateFactory;
import burlap.oomdp.statehashing.SimpleHashableStateFactory;
import burlap.oomdp.visualizer.Visualizer;

public class BasicBehavior {


	GridWorldDomain gwdg;
	Domain domain;
	RewardFunction rf;
	TerminalFunction tf;
	StateConditionTest l0sc;
	State initialState;
	HashableStateFactory hashingFactory;
	Environment env;

	public BasicBehavior() {
		gwdg = new GridWorldDomain(11, 11);
		gwdg.setMapToFourRooms();
		//domain  = gwdg.generateDomain();
		Domain d0 = gwdg.generateDomain();

		rf = new UniformCostRF();
		tf = new SinglePFTF(d0.getPropFunction(GridWorldDomain.PFATLOCATION));
		l0sc = new TFGoalCondition(tf);

		initialState = GridWorldDomain.getOneAgentOneLocationState(d0);
		GridWorldDomain.setAgent(initialState, 0, 0);
		GridWorldDomain.setLocation(initialState, 0, 10, 10);

		hashingFactory = new SimpleHashableStateFactory();

		env = new SimulatedEnvironment(d0, rf, tf, initialState);

		StateConditionTest l0sc = new L0Goal(initialState, d0.getPropFunction(GridWorldDomain.PFATLOCATION));
		RewardFunction rfl0 = new GoalBasedRF(l0sc,1.);
		TerminalFunction tfl0 = new GoalConditionTF(l0sc);

		List<Domain> domainList = new ArrayList<Domain>();
		domainList.add(0,d0);

		List<AMDPPolicyGenerator> pgList = new ArrayList<AMDPPolicyGenerator>();
		pgList.add(0,new l0PolicyGenerator(d0));

		AMDPAgent agent = new AMDPAgent(domainList, pgList,rfl0,tfl0);

	}

	public class L0Goal implements StateConditionTest {
		State initialstate;
		PropositionalFunction pf;

		public L0Goal(State initial, PropositionalFunction goalLocation) {
			this.initialstate = initial ;
			this.pf = goalLocation;

		}
		@Override
		public boolean satisfies(State s){
			ObjectInstance agent =s.getObject(GridWorldDomain.CLASSAGENT);
			double x =agent.getRealValForAttribute(GridWorldDomain.ATTX);
			double y =agent.getRealValForAttribute(GridWorldDomain.ATTY);
			ObjectInstance goal = s.getObject(GridWorldDomain.CLASSLOCATION);
			double xL = goal.getRealValForAttribute(GridWorldDomain.ATTX);
			double yL = goal.getRealValForAttribute(GridWorldDomain.ATTY);


			if(x == xL && y == yL){
				return true;
			}

			return false;
		}
	}


	public class l0PolicyGenerator implements AMDPPolicyGenerator{

		private Domain l0;
		public l0PolicyGenerator(Domain l0ln){
			l0 = l0ln;
		}
		public Policy generatePolicy(State initialState, RewardFunction rf, TerminalFunction tf){
			Planner planner = new ValueIteration(domain, rf, tf, 0.99, hashingFactory, 0.001, 100);
			Policy p = planner.planFromState(initialState);
//			System.out.println(s.toString());
			return p;
		}
	}

	public void visualize(String outputPath){
		Visualizer v = GridWorldVisualizer.getVisualizer(gwdg.getMap());
		new EpisodeSequenceVisualizer(v,domain,outputPath);
	}

	public void BFSExample(String outputPath){

		DeterministicPlanner planner = new BFS(domain, l0sc, hashingFactory);
		Policy p = planner.planFromState(initialState);
		p.evaluateBehavior(initialState, rf, tf).writeToFile(outputPath + "bfs");

	}

	public void DFSExample(String outputPath){

		DeterministicPlanner planner = new DFS(domain, l0sc, hashingFactory);
		Policy p = planner.planFromState(initialState);
		p.evaluateBehavior(initialState, rf, tf).writeToFile(outputPath + "dfs");

	}

	public void AStarExample(String outputPath){

		Heuristic mdistHeuristic = new Heuristic() {
			@Override
			public double h(State s) {

				ObjectInstance agent = s.getFirstObjectOfClass(GridWorldDomain.CLASSAGENT);
				ObjectInstance location = s.getFirstObjectOfClass(GridWorldDomain.CLASSLOCATION);

				int ax = agent.getIntValForAttribute(GridWorldDomain.ATTX);
				int ay = agent.getIntValForAttribute(GridWorldDomain.ATTY);

				int lx = location.getIntValForAttribute(GridWorldDomain.ATTX);
				int ly = location.getIntValForAttribute(GridWorldDomain.ATTY);

				double mdist = Math.abs(ax-lx) + Math.abs(ay-ly);

				return -mdist;
			}
		};

		DeterministicPlanner planner = new AStar(domain, rf, l0sc, hashingFactory, mdistHeuristic);
		Policy p = planner.planFromState(initialState);

		p.evaluateBehavior(initialState, rf, tf).writeToFile(outputPath + "astar");

	}

	public void valueIterationExample(String outputPath){

		Planner planner = new ValueIteration(domain, rf, tf, 0.99, hashingFactory, 0.001, 100);
		Policy p = planner.planFromState(initialState);

		p.evaluateBehavior(initialState, rf, tf).writeToFile(outputPath + "vi");

		//simpleValueFunctionVis((ValueFunction)planner, p);
		manualValueFunctionVis((ValueFunction)planner, p);

	}


	public void qLearningExample(String outputPath){

		LearningAgent agent = new QLearning(domain, 0.99, hashingFactory, 0., 1.);

		//run learning for 50 episodes
		for(int i = 0; i < 50; i++){
			EpisodeAnalysis ea = agent.runLearningEpisode(env);

			ea.writeToFile(outputPath + "ql_" + i);
			System.out.println(i + ": " + ea.maxTimeStep());

			//reset environment for next learning episode
			env.resetEnvironment();
		}

	}


	public void sarsaLearningExample(String outputPath){

		LearningAgent agent = new SarsaLam(domain, 0.99, hashingFactory, 0., 0.5, 0.3);

		//run learning for 50 episodes
		for(int i = 0; i < 50; i++){
			EpisodeAnalysis ea = agent.runLearningEpisode(env);

			ea.writeToFile(outputPath + "sarsa_" + i);
			System.out.println(i + ": " + ea.maxTimeStep());

			//reset environment for next learning episode
			env.resetEnvironment();
		}

	}

	public void simpleValueFunctionVis(ValueFunction valueFunction, Policy p){

		List<State> allStates = StateReachability.getReachableStates(initialState, 
									(SADomain)domain, hashingFactory);
		ValueFunctionVisualizerGUI gui = GridWorldDomain.getGridWorldValueFunctionVisualization(
											allStates, valueFunction, p);
		gui.initGUI();

	}

	public void manualValueFunctionVis(ValueFunction valueFunction, Policy p){

		List<State> allStates = StateReachability.getReachableStates(initialState, 
									(SADomain)domain, hashingFactory);

		//define color function
		LandmarkColorBlendInterpolation rb = new LandmarkColorBlendInterpolation();
		rb.addNextLandMark(0., Color.RED);
		rb.addNextLandMark(1., Color.BLUE);

		//define a 2D painter of state values, specifying which attributes correspond 
		//to the x and y coordinates of the canvas
		StateValuePainter2D svp = new StateValuePainter2D(rb);
		svp.setXYAttByObjectClass(GridWorldDomain.CLASSAGENT, GridWorldDomain.ATTX,
				GridWorldDomain.CLASSAGENT, GridWorldDomain.ATTY);


		//create our ValueFunctionVisualizer that paints for all states
		//using the ValueFunction source and the state value painter we defined
		ValueFunctionVisualizerGUI gui = new ValueFunctionVisualizerGUI(
												allStates, svp, valueFunction);

		//define a policy painter that uses arrow glyphs for each of the grid world actions
		PolicyGlyphPainter2D spp = new PolicyGlyphPainter2D();
		spp.setXYAttByObjectClass(GridWorldDomain.CLASSAGENT, GridWorldDomain.ATTX,
				GridWorldDomain.CLASSAGENT, GridWorldDomain.ATTY);
		spp.setActionNameGlyphPainter(GridWorldDomain.ACTIONNORTH, new ArrowActionGlyph(0));
		spp.setActionNameGlyphPainter(GridWorldDomain.ACTIONSOUTH, new ArrowActionGlyph(1));
		spp.setActionNameGlyphPainter(GridWorldDomain.ACTIONEAST, new ArrowActionGlyph(2));
		spp.setActionNameGlyphPainter(GridWorldDomain.ACTIONWEST, new ArrowActionGlyph(3));
		spp.setRenderStyle(PolicyGlyphPainter2D.PolicyGlyphRenderStyle.DISTSCALED);


		//add our policy renderer to it
		gui.setSpp(spp);
		gui.setPolicy(p);

		//set the background color for places where states are not rendered to grey
		gui.setBgColor(Color.GRAY);

		//start it
		gui.initGUI();



	}


	public void experimentAndPlotter(){

		//different reward function for more interesting results
		((SimulatedEnvironment)env).setRf(new GoalBasedRF(this.l0sc, 5.0, -0.1));

		/**
		 * Create factories for Q-learning agent and SARSA agent to compare
		 */
		LearningAgentFactory qLearningFactory = new LearningAgentFactory() {
			@Override
			public String getAgentName() {
				return "Q-Learning";
			}

			@Override
			public LearningAgent generateAgent() {
				return new QLearning(domain, 0.99, hashingFactory, 0.3, 0.1);
			}
		};

		LearningAgentFactory sarsaLearningFactory = new LearningAgentFactory() {
			@Override
			public String getAgentName() {
				return "SARSA";
			}

			@Override
			public LearningAgent generateAgent() {
				return new SarsaLam(domain, 0.99, hashingFactory, 0.0, 0.1, 1.);
			}
		};

		LearningAlgorithmExperimenter exp = new LearningAlgorithmExperimenter(env, 10, 100, 
												qLearningFactory, sarsaLearningFactory);
		exp.setUpPlottingConfiguration(500, 250, 2, 1000,
				TrialMode.MOSTRECENTANDAVERAGE,
				PerformanceMetric.CUMULATIVESTEPSPEREPISODE,
				PerformanceMetric.AVERAGEEPISODEREWARD);

		exp.startExperiment();
		exp.writeStepAndEpisodeDataToCSV("expData");

	}


	public static void main(String[] args) {

		BasicBehavior example = new BasicBehavior();
		String outputPath = "output/";

		//example.BFSExample(outputPath);
		example.DFSExample(outputPath);
		//example.AStarExample(outputPath);
		//example.valueIterationExample(outputPath);
		//example.qLearningExample(outputPath);
		//example.sarsaLearningExample(outputPath);

		//example.experimentAndPlotter();

		example.visualize(outputPath);

	
	
}

}
