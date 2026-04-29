"""
Mitigation Module
"""

import logging
from datetime import datetime
from pathlib import Path
import json

logger = logging.getLogger(__name__)

class MitigationEngine:
    def __init__(self):
        self.mitigation_log = []
        self.node_status = {}
        self.reputation_scores = {}
        self.attack_counter = {}

    def get_mitigation_action(self, attack_type, confidence, node_id):
        """Determine mitigation action based on attack type"""

        actions = {
            0: {'action': 'none', 'desc': 'Normal traffic', 'severity': 'low'},
            1: {'action': 'isolate', 'desc': 'Isolating node - Blackhole attack', 'severity': 'critical'},
            2: {'action': 'throttle', 'desc': 'Throttling node - Grayhole attack', 'severity': 'high'},
            3: {'action': 'rate_limit', 'desc': 'Rate limiting - Flooding attack', 'severity': 'high'},
            4: {'action': 'validate', 'desc': 'Identity validation - Sybil attack', 'severity': 'critical'},
            5: {'action': 'reroute', 'desc': 'Rerouting traffic - Sinkhole attack', 'severity': 'high'},
            6: {'action': 'reschedule', 'desc': 'Resetting TDMA schedule', 'severity': 'medium'},
            7: {'action': 'filter', 'desc': 'Filtering hello messages', 'severity': 'medium'}
        }

        action = actions.get(attack_type, actions[0])

        # Check confidence threshold
        if confidence < 0.7:
            return {'action': 'monitor', 'desc': 'Monitoring only - low confidence', 'severity': 'low'}

        # Update counters
        if node_id:
            self.attack_counter[node_id] = self.attack_counter.get(node_id, 0) + 1
            self.reputation_scores[node_id] = max(0, 100 - (self.attack_counter[node_id] * 20))

            if self.attack_counter[node_id] > 2:
                action['severity'] = 'critical'
                action['desc'] += ' - Repeated attacks'

        return action

    def execute_mitigation(self, action, node_id, attack_type):
        """Execute mitigation action"""
        response = {
            'timestamp': datetime.now().isoformat(),
            'node_id': node_id,
            'attack_type': attack_type,
            'action': action['action'],
            'description': action['desc'],
            'severity': action['severity'],
            'status': 'executed'
        }

        # Update node status
        if action['action'] == 'isolate':
            self.node_status[node_id] = 'isolated'
        elif action['action'] in ['throttle', 'rate_limit']:
            self.node_status[node_id] = 'restricted'
        else:
            self.node_status[node_id] = 'monitored'

        self.mitigation_log.append(response)
        self._save_log()

        return response

    def get_node_status(self, node_id):
        """Get current status of a node"""
        return {
            'node_id': node_id,
            'status': self.node_status.get(node_id, 'active'),
            'reputation': self.reputation_scores.get(node_id, 100),
            'attack_count': self.attack_counter.get(node_id, 0)
        }

    def _save_log(self):
        """Save mitigation log"""
        Path("logs").mkdir(exist_ok=True)
        with open('logs/mitigation.json', 'w') as f:
            json.dump(self.mitigation_log[-100:], f, indent=2)