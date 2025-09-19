"""
State Predictor Module

Implements preemptive state prediction and distribution for demonstrating
"negative latency" through temporal coordination. Predicts future interface
states and user interactions to enable preemptive content delivery.

This module showcases how Sango Rine Shumba's temporal coordination can
enable revolutionary improvements in user experience responsiveness.
"""

import asyncio
import time
import random
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import logging
from collections import deque
import uuid

@dataclass
class UserAction:
    """Represents a predicted user action"""
    
    action_id: str
    action_type: str  # 'click', 'scroll', 'navigation', 'input', 'gesture'
    predicted_time: float
    confidence: float
    
    # Action parameters
    target_element: Optional[str] = None
    coordinates: Optional[Tuple[float, float]] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Context information
    current_state_id: str = ""
    session_context: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize action properties"""
        self.age_seconds = 0.0
        self.was_correct = None  # Will be set when actual action occurs
        self.actual_time: Optional[float] = None
    
    @property
    def prediction_error(self) -> Optional[float]:
        """Calculate prediction error if actual action occurred"""
        if self.actual_time is None:
            return None
        return abs(self.actual_time - self.predicted_time)
    
    @property
    def is_overdue(self) -> bool:
        """Check if predicted action is overdue"""
        return time.time() > self.predicted_time + 2.0  # 2 second tolerance

@dataclass
class InterfaceState:
    """Represents an interface state for preemptive distribution"""
    
    state_id: str
    application_context: str
    state_data: Dict[str, Any]
    generation_time: float
    
    # State properties
    complexity_score: float = 0.0
    rendering_time_ms: float = 0.0
    dependencies: List[str] = field(default_factory=list)
    cache_priority: int = 0
    
    # Prediction metadata
    triggering_action: Optional[UserAction] = None
    prediction_confidence: float = 0.0
    temporal_coordinate: float = 0.0
    
    def __post_init__(self):
        """Initialize state properties"""
        self.is_preemptive = True
        self.hit_count = 0
        self.miss_count = 0
        
        if self.complexity_score == 0.0:
            self.complexity_score = self._calculate_complexity()
    
    def _calculate_complexity(self) -> float:
        """Calculate state complexity based on data size and structure"""
        data_size = len(json.dumps(self.state_data))
        complexity = min(1.0, data_size / 10000)  # Normalize to 1.0 for 10KB
        
        # Add complexity for nested structures
        if isinstance(self.state_data, dict):
            nested_levels = self._count_nested_levels(self.state_data)
            complexity += min(0.5, nested_levels * 0.1)
        
        return complexity
    
    def _count_nested_levels(self, obj, level=0) -> int:
        """Count nested levels in data structure"""
        if not isinstance(obj, dict):
            return level
        
        max_level = level
        for value in obj.values():
            if isinstance(value, dict):
                max_level = max(max_level, self._count_nested_levels(value, level + 1))
        
        return max_level
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate"""
        total = self.hit_count + self.miss_count
        return self.hit_count / total if total > 0 else 0.0

@dataclass
class PredictionSession:
    """Represents a user session for state prediction"""
    
    session_id: str
    user_context: Dict[str, Any]
    start_time: float
    
    # Session state
    current_state: Optional[InterfaceState] = None
    action_history: List[UserAction] = field(default_factory=list)
    state_cache: Dict[str, InterfaceState] = field(default_factory=dict)
    
    # Performance metrics
    predictions_made: int = 0
    predictions_correct: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    
    def __post_init__(self):
        """Initialize session"""
        self.last_activity_time = self.start_time
        self.is_active = True
    
    @property
    def prediction_accuracy(self) -> float:
        """Calculate prediction accuracy"""
        return self.predictions_correct / max(1, self.predictions_made)
    
    @property
    def session_duration(self) -> float:
        """Get session duration in seconds"""
        return time.time() - self.start_time

class StatePredictor:
    """
    Preemptive state prediction engine
    
    Implements sophisticated user interaction prediction and preemptive
    state generation to demonstrate "negative latency" capabilities
    of the Sango Rine Shumba framework.
    
    Features:
    - User behavior pattern learning
    - Temporal coordination-aware state prediction
    - Preemptive state caching and distribution
    - Real-time accuracy monitoring
    """
    
    def __init__(self, precision_calculator, data_collector=None):
        """Initialize state predictor"""
        self.precision_calculator = precision_calculator
        self.data_collector = data_collector
        self.logger = logging.getLogger(__name__)
        
        # Prediction models
        self.user_models: Dict[str, Dict[str, Any]] = {}
        self.interaction_patterns: Dict[str, List[UserAction]] = {}
        
        # Session management
        self.active_sessions: Dict[str, PredictionSession] = {}
        self.state_cache: Dict[str, InterfaceState] = {}
        
        # Prediction parameters
        self.prediction_horizon_ms = 1000  # 1 second default
        self.confidence_threshold = 0.7
        self.cache_size_limit = 1000
        
        # Performance tracking
        self.performance_metrics = {
            'total_predictions': 0,
            'correct_predictions': 0,
            'states_generated': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'average_prediction_time': 0.0,
            'user_experience_improvement': 0.0,
            'accuracy_history': deque(maxlen=1000)
        }
        
        # Background prediction patterns (for demo purposes)
        self.demo_patterns = self._initialize_demo_patterns()
        
        # Service state
        self.is_running = False
        
        self.logger.info("State predictor initialized")
    
    def _initialize_demo_patterns(self) -> Dict[str, Any]:
        """Initialize demonstration user interaction patterns"""
        return {
            'web_browsing': {
                'click_probability': 0.6,
                'scroll_probability': 0.8,
                'navigation_probability': 0.2,
                'average_dwell_time': 3.0,
                'common_sequences': [
                    ['scroll', 'click', 'scroll'],
                    ['click', 'navigation', 'scroll'],
                    ['scroll', 'scroll', 'click']
                ]
            },
            'form_interaction': {
                'input_probability': 0.9,
                'submit_probability': 0.3,
                'validation_delay': 0.5,
                'error_probability': 0.1,
                'common_sequences': [
                    ['input', 'input', 'submit'],
                    ['input', 'click', 'input'],
                    ['input', 'input', 'input', 'submit']
                ]
            },
            'media_consumption': {
                'play_probability': 0.95,
                'pause_probability': 0.4,
                'seek_probability': 0.3,
                'quality_change_probability': 0.1,
                'common_sequences': [
                    ['play', 'pause', 'play'],
                    ['play', 'seek', 'play'],
                    ['play', 'play', 'pause']
                ]
            }
        }
    
    async def start_prediction_service(self):
        """Start preemptive state prediction service"""
        self.logger.info("Starting preemptive state prediction service...")
        self.is_running = True
        
        # Start background tasks
        prediction_task = asyncio.create_task(self._prediction_loop())
        session_management_task = asyncio.create_task(self._session_management_loop())
        cache_cleanup_task = asyncio.create_task(self._cache_cleanup_loop())
        
        try:
            await asyncio.gather(prediction_task, session_management_task, cache_cleanup_task)
        except asyncio.CancelledError:
            self.logger.info("State prediction service stopped")
    
    async def _prediction_loop(self):
        """Main prediction loop"""
        while self.is_running:
            try:
                prediction_start = time.time()
                
                # Generate predictions for all active sessions
                for session in self.active_sessions.values():
                    if session.is_active and session.current_state:
                        await self._generate_session_predictions(session)
                
                # Update prediction timing metrics
                prediction_time = time.time() - prediction_start
                self.performance_metrics['average_prediction_time'] = (
                    0.9 * self.performance_metrics['average_prediction_time'] +
                    0.1 * prediction_time
                )
                
                await asyncio.sleep(0.2)  # 200ms prediction interval
                
            except Exception as e:
                self.logger.error(f"Error in prediction loop: {e}")
                await asyncio.sleep(1.0)
    
    async def _generate_session_predictions(self, session: PredictionSession):
        """Generate predictions for a specific session"""
        try:
            current_time = time.time()
            
            # Get prediction context
            context = self._build_prediction_context(session)
            
            # Generate user action predictions
            predicted_actions = await self._predict_user_actions(session, context)
            
            # Generate preemptive states for predicted actions
            for action in predicted_actions:
                if action.confidence >= self.confidence_threshold:
                    preemptive_state = await self._generate_preemptive_state(session, action)
                    if preemptive_state:
                        # Cache preemptive state
                        self._cache_preemptive_state(session, preemptive_state)
                        
                        self.performance_metrics['states_generated'] += 1
            
            session.predictions_made += len(predicted_actions)
            self.performance_metrics['total_predictions'] += len(predicted_actions)
            
        except Exception as e:
            self.logger.error(f"Error generating session predictions: {e}")
    
    def _build_prediction_context(self, session: PredictionSession) -> Dict[str, Any]:
        """Build context for prediction generation"""
        current_time = time.time()
        
        # Get temporal coordination info
        coordination_matrix = self.precision_calculator.get_current_coordination_matrix()
        
        context = {
            'current_time': current_time,
            'session_duration': session.session_duration,
            'recent_actions': session.action_history[-10:],  # Last 10 actions
            'current_state': session.current_state,
            'user_context': session.user_context,
            'temporal_coordination': {
                'matrix_available': coordination_matrix is not None,
                'sync_quality': coordination_matrix.synchronization_quality if coordination_matrix else 0.0,
                'temporal_window': coordination_matrix.temporal_window_duration if coordination_matrix else 0.0
            }
        }
        
        return context
    
    async def _predict_user_actions(self, session: PredictionSession, context: Dict[str, Any]) -> List[UserAction]:
        """Predict future user actions based on context"""
        predictions = []
        current_time = context['current_time']
        
        # Determine user behavior pattern
        pattern_type = self._classify_user_behavior(session, context)
        pattern_data = self.demo_patterns.get(pattern_type, self.demo_patterns['web_browsing'])
        
        # Generate action predictions based on pattern
        for action_type, probability in [
            ('click', pattern_data['click_probability']),
            ('scroll', pattern_data['scroll_probability']),
            ('navigation', pattern_data.get('navigation_probability', 0.2))
        ]:
            if random.random() < probability:
                # Calculate prediction time within horizon
                prediction_time = current_time + random.uniform(0.1, self.prediction_horizon_ms / 1000)
                
                # Calculate confidence based on pattern strength and temporal coordination
                base_confidence = probability
                temporal_boost = context['temporal_coordination']['sync_quality'] * 0.2
                confidence = min(1.0, base_confidence + temporal_boost)
                
                action = UserAction(
                    action_id=str(uuid.uuid4()),
                    action_type=action_type,
                    predicted_time=prediction_time,
                    confidence=confidence,
                    current_state_id=session.current_state.state_id if session.current_state else "",
                    session_context=context['user_context']
                )
                
                predictions.append(action)
        
        return predictions
    
    def _classify_user_behavior(self, session: PredictionSession, context: Dict[str, Any]) -> str:
        """Classify user behavior pattern for prediction"""
        
        # Simple classification based on session context
        if 'form' in context['user_context'].get('page_type', '').lower():
            return 'form_interaction'
        elif 'video' in context['user_context'].get('content_type', '').lower():
            return 'media_consumption'
        else:
            return 'web_browsing'
    
    async def _generate_preemptive_state(self, session: PredictionSession, action: UserAction) -> Optional[InterfaceState]:
        """Generate preemptive interface state for predicted action"""
        try:
            # Get temporal coordinate for state generation
            coordination_matrix = self.precision_calculator.get_current_coordination_matrix()
            temporal_coordinate = 0.0
            
            if coordination_matrix:
                # Use temporal window for state coordination
                progress = min(1.0, action.confidence)  # Higher confidence = later in window
                temporal_coordinate = coordination_matrix.get_temporal_coordinate(progress)
            
            # Generate state based on action type
            state_data = self._generate_state_data(session, action)
            
            preemptive_state = InterfaceState(
                state_id=f"preemptive_{action.action_id}",
                application_context=session.user_context.get('application', 'demo_app'),
                state_data=state_data,
                generation_time=time.time(),
                triggering_action=action,
                prediction_confidence=action.confidence,
                temporal_coordinate=temporal_coordinate
            )
            
            return preemptive_state
            
        except Exception as e:
            self.logger.error(f"Error generating preemptive state: {e}")
            return None
    
    def _generate_state_data(self, session: PredictionSession, action: UserAction) -> Dict[str, Any]:
        """Generate realistic state data based on predicted action"""
        
        base_state = {
            'timestamp': time.time(),
            'session_id': session.session_id,
            'action_type': action.action_type,
            'confidence': action.confidence
        }
        
        # Generate action-specific state data
        if action.action_type == 'click':
            base_state.update({
                'content': f"Content for click action {action.action_id[:8]}",
                'ui_elements': [
                    {'type': 'button', 'id': 'btn_primary', 'state': 'active'},
                    {'type': 'text', 'id': 'result_text', 'content': 'Processing...'}
                ],
                'loading_states': ['fetch_data', 'render_ui', 'update_display']
            })
        
        elif action.action_type == 'scroll':
            base_state.update({
                'viewport': {'top': random.randint(0, 2000), 'height': 800},
                'visible_elements': [f"element_{i}" for i in range(10)],
                'lazy_load_triggered': random.choice([True, False])
            })
        
        elif action.action_type == 'navigation':
            base_state.update({
                'target_page': f"page_{random.randint(1, 10)}",
                'navigation_type': 'client_side',
                'preload_resources': [
                    f"resource_{i}.js" for i in range(3)
                ]
            })
        
        # Add complexity based on user context
        if session.user_context.get('user_level') == 'power_user':
            base_state['advanced_features'] = {
                'keyboard_shortcuts': True,
                'batch_operations': True,
                'custom_ui': True
            }
        
        return base_state
    
    def _cache_preemptive_state(self, session: PredictionSession, state: InterfaceState):
        """Cache preemptive state for future use"""
        
        # Add to session cache
        session.state_cache[state.state_id] = state
        
        # Add to global cache
        self.state_cache[state.state_id] = state
        
        # Enforce cache size limits
        if len(self.state_cache) > self.cache_size_limit:
            # Remove oldest states
            oldest_states = sorted(self.state_cache.items(), key=lambda x: x[1].generation_time)
            for state_id, _ in oldest_states[:len(self.state_cache) - self.cache_size_limit]:
                del self.state_cache[state_id]
    
    async def _session_management_loop(self):
        """Manage active sessions and simulate user interactions"""
        while self.is_running:
            try:
                current_time = time.time()
                
                # Create demo sessions if none exist
                if len(self.active_sessions) < 5:  # Maintain 5 demo sessions
                    await self._create_demo_session()
                
                # Simulate user interactions for demo
                for session in list(self.active_sessions.values()):
                    if session.is_active:
                        await self._simulate_user_interaction(session)
                
                # Clean up inactive sessions
                inactive_sessions = [
                    sid for sid, session in self.active_sessions.items()
                    if current_time - session.last_activity_time > 300  # 5 minutes
                ]
                
                for session_id in inactive_sessions:
                    self.active_sessions[session_id].is_active = False
                    del self.active_sessions[session_id]
                
                await asyncio.sleep(2.0)  # Session management every 2 seconds
                
            except Exception as e:
                self.logger.error(f"Error in session management: {e}")
                await asyncio.sleep(5.0)
    
    async def _create_demo_session(self):
        """Create a demonstration session"""
        session_id = str(uuid.uuid4())
        
        # Generate realistic user context
        user_contexts = [
            {
                'user_id': f'demo_user_{len(self.active_sessions) + 1}',
                'user_level': random.choice(['beginner', 'intermediate', 'power_user']),
                'application': 'web_app',
                'page_type': random.choice(['homepage', 'form', 'content', 'dashboard']),
                'device_type': random.choice(['desktop', 'mobile', 'tablet']),
                'connection_quality': random.choice(['high', 'medium', 'low'])
            }
        ]
        
        session = PredictionSession(
            session_id=session_id,
            user_context=random.choice(user_contexts),
            start_time=time.time()
        )
        
        # Create initial state
        session.current_state = InterfaceState(
            state_id=f"initial_{session_id[:8]}",
            application_context=session.user_context['application'],
            state_data={'initial': True, 'loaded': True},
            generation_time=time.time()
        )
        
        self.active_sessions[session_id] = session
        self.logger.debug(f"Created demo session {session_id}")
    
    async def _simulate_user_interaction(self, session: PredictionSession):
        """Simulate user interactions for demonstration purposes"""
        current_time = time.time()
        
        # Check if it's time for a simulated action
        if current_time - session.last_activity_time < random.uniform(1, 5):
            return  # Not time for action yet
        
        # Simulate user action
        action_types = ['click', 'scroll', 'navigation']
        action_type = random.choice(action_types)
        
        # Check if we predicted this action
        predicted_action = None
        for action in session.action_history[-10:]:  # Check recent predictions
            if (action.action_type == action_type and 
                abs(current_time - action.predicted_time) < 1.0):  # Within 1 second
                predicted_action = action
                break
        
        # Update prediction accuracy
        if predicted_action:
            predicted_action.was_correct = True
            predicted_action.actual_time = current_time
            session.predictions_correct += 1
            self.performance_metrics['correct_predictions'] += 1
            
            # Check if we have a cached state for this action
            state_key = f"preemptive_{predicted_action.action_id}"
            if state_key in session.state_cache:
                session.cache_hits += 1
                self.performance_metrics['cache_hits'] += 1
                
                # Use cached state
                session.current_state = session.state_cache[state_key]
                session.current_state.hit_count += 1
                
                self.logger.debug(f"Cache hit for predicted action {action_type}")
            else:
                session.cache_misses += 1
                self.performance_metrics['cache_misses'] += 1
        
        # Create actual action record
        actual_action = UserAction(
            action_id=str(uuid.uuid4()),
            action_type=action_type,
            predicted_time=current_time,
            confidence=1.0,  # Actual actions have 100% confidence
            current_state_id=session.current_state.state_id if session.current_state else ""
        )
        actual_action.actual_time = current_time
        
        session.action_history.append(actual_action)
        session.last_activity_time = current_time
        
        # Update accuracy tracking
        accuracy = session.prediction_accuracy
        self.performance_metrics['accuracy_history'].append(accuracy)
        
        # Calculate user experience improvement
        if predicted_action and predicted_action.was_correct:
            # Simulated improvement from preemptive loading
            improvement = random.uniform(0.3, 0.8)  # 30-80% improvement
            self.performance_metrics['user_experience_improvement'] = (
                0.9 * self.performance_metrics['user_experience_improvement'] +
                0.1 * improvement
            )
    
    async def _cache_cleanup_loop(self):
        """Clean up expired cache entries"""
        while self.is_running:
            try:
                current_time = time.time()
                
                # Clean up expired states
                expired_states = [
                    state_id for state_id, state in self.state_cache.items()
                    if current_time - state.generation_time > 30.0  # 30 second expiry
                ]
                
                for state_id in expired_states:
                    del self.state_cache[state_id]
                
                # Clean up session caches
                for session in self.active_sessions.values():
                    expired_session_states = [
                        state_id for state_id, state in session.state_cache.items()
                        if current_time - state.generation_time > 30.0
                    ]
                    
                    for state_id in expired_session_states:
                        del session.state_cache[state_id]
                
                if expired_states:
                    self.logger.debug(f"Cleaned up {len(expired_states)} expired cache entries")
                
                await asyncio.sleep(10.0)  # Cleanup every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Error in cache cleanup: {e}")
                await asyncio.sleep(30.0)
    
    def get_prediction_statistics(self) -> Dict[str, Any]:
        """Get comprehensive prediction statistics"""
        
        # Calculate overall accuracy
        overall_accuracy = (self.performance_metrics['correct_predictions'] / 
                          max(1, self.performance_metrics['total_predictions']))
        
        # Calculate cache hit rate
        total_cache_requests = self.performance_metrics['cache_hits'] + self.performance_metrics['cache_misses']
        cache_hit_rate = (self.performance_metrics['cache_hits'] / 
                         max(1, total_cache_requests))
        
        # Session statistics
        active_session_count = len(self.active_sessions)
        session_accuracies = [s.prediction_accuracy for s in self.active_sessions.values()]
        avg_session_accuracy = sum(session_accuracies) / max(1, len(session_accuracies))
        
        return {
            'active_sessions': active_session_count,
            'total_predictions': self.performance_metrics['total_predictions'],
            'correct_predictions': self.performance_metrics['correct_predictions'],
            'overall_accuracy': overall_accuracy,
            'average_session_accuracy': avg_session_accuracy,
            'states_generated': self.performance_metrics['states_generated'],
            'cache_hits': self.performance_metrics['cache_hits'],
            'cache_misses': self.performance_metrics['cache_misses'],
            'cache_hit_rate': cache_hit_rate,
            'cached_states': len(self.state_cache),
            'average_prediction_time': self.performance_metrics['average_prediction_time'],
            'user_experience_improvement': self.performance_metrics['user_experience_improvement'],
            'recent_accuracy_trend': list(self.performance_metrics['accuracy_history'])[-20:]  # Last 20 measurements
        }
    
    def stop(self):
        """Stop state prediction service"""
        self.is_running = False
        self.logger.info("State predictor stopped")
