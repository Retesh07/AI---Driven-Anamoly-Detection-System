"""
Statistical visualization for threat detection outputs.
Generates graphs for weapon and loitering detection patterns.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from collections import defaultdict


def generate_weapon_statistics_graph(timeline, output_path, title="Weapon Detection Statistics"):
    """
    Generate graphs showing weapon detection patterns over time.
    
    Args:
        timeline: List of timeline entries with frame-by-frame data
        output_path: Path to save the visualization
        title: Graph title
    """
    if not timeline:
        return
    
    frames = []
    weapon_detected_frames = []
    gun_detected_frames = []
    weapon_scores = []
    threats_with_weapons = []
    
    for entry in timeline:
        frame_num = entry['frame']
        frames.append(frame_num)
        
        weapon_count = 0
        gun_count = 0
        max_weapon_score = 0.0
        
        for person in entry.get('persons', []):
            if person.get('weapon_present', False):
                weapon_count += 1
                max_weapon_score = max(max_weapon_score, person.get('weapon_score', 0.0))
                
                if person.get('is_gun', False):
                    gun_count += 1
        
        weapon_detected_frames.append(weapon_count > 0)
        gun_detected_frames.append(gun_count > 0)
        weapon_scores.append(max_weapon_score)
        threats_with_weapons.append(weapon_count)
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # ===== SUBPLOT 1: Weapon Detection Timeline =====
    ax1 = axes[0]
    
    # Fill areas for weapon detection
    ax1.fill_between(frames, 0, 1, where=weapon_detected_frames, 
                     alpha=0.3, color='orange', label='Weapon Detected')
    ax1.fill_between(frames, 0, 1, where=gun_detected_frames, 
                     alpha=0.5, color='red', label='Gun Detected')
    
    # Mark gun detections with vertical lines
    for i, (frame, is_gun) in enumerate(zip(frames, gun_detected_frames)):
        if is_gun:
            ax1.axvline(frame, color='darkred', linewidth=0.8, alpha=0.6, linestyle='--')
    
    ax1.set_xlim(frames[0], frames[-1])
    ax1.set_ylim(0, 1.1)
    ax1.set_ylabel('Weapon Presence')
    ax1.set_title(f'{title} - Detection Timeline')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.2)
    
    # ===== SUBPLOT 2: Weapon Confidence Scores =====
    ax2 = axes[1]
    
    # Plot weapon scores as line
    ax2.plot(frames, weapon_scores, color='orange', linewidth=2, label='Max Weapon Score')
    ax2.fill_between(frames, 0, weapon_scores, alpha=0.3, color='orange')
    
    # Highlight high confidence regions
    for i in range(len(frames)):
        if weapon_scores[i] > 0.5:
            ax2.axvspan(frames[i]-2, frames[i]+2, alpha=0.1, color='red')
    
    ax2.set_xlim(frames[0], frames[-1])
    ax2.set_ylim(0, 1.0)
    ax2.set_ylabel('Weapon Confidence Score')
    ax2.set_title('Weapon Detection Confidence')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.2)
    
    # ===== SUBPLOT 3: Persons with Weapons =====
    ax3 = axes[2]
    
    # Bar chart of person count with weapons
    colors = ['red' if count > 0 else 'lightgray' for count in threats_with_weapons]
    ax3.bar(frames, threats_with_weapons, color=colors, alpha=0.7, width=1)
    ax3.set_xlim(frames[0], frames[-1])
    ax3.set_ylim(0, max(threats_with_weapons) + 1 if threats_with_weapons else 1)
    ax3.set_xlabel('Frame')
    ax3.set_ylabel('Number of Persons with Weapons')
    ax3.set_title('Person Count - Armed Individuals')
    ax3.grid(True, alpha=0.2, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=130, bbox_inches='tight')
    plt.close()


def generate_loitering_statistics_graph(timeline, output_path, title="Loitering Detection Statistics"):
    """
    Generate graphs showing loitering detection patterns over time.
    
    Args:
        timeline: List of timeline entries with frame-by-frame data
        output_path: Path to save the visualization
        title: Graph title
    """
    if not timeline:
        return
    
    frames = []
    loitering_detected_frames = []
    loitering_scores = []
    persons_loitering = []
    
    for entry in timeline:
        frame_num = entry['frame']
        frames.append(frame_num)
        
        loitering_count = 0
        max_loitering_score = 0.0
        
        for person in entry.get('persons', []):
            if person.get('loitering_detected', False):
                loitering_count += 1
                max_loitering_score = max(max_loitering_score, person.get('loitering_score', 0.0))
        
        loitering_detected_frames.append(loitering_count > 0)
        loitering_scores.append(max_loitering_score)
        persons_loitering.append(loitering_count)
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # ===== SUBPLOT 1: Loitering Detection Timeline =====
    ax1 = axes[0]
    
    # Fill areas for loitering detection
    ax1.fill_between(frames, 0, 1, where=loitering_detected_frames, 
                     alpha=0.4, color='purple', label='Loitering Detected')
    
    # Mark loitering detections with vertical lines
    for i, (frame, is_loitering) in enumerate(zip(frames, loitering_detected_frames)):
        if is_loitering:
            ax1.axvline(frame, color='purple', linewidth=0.8, alpha=0.5, linestyle=':')
    
    ax1.set_xlim(frames[0], frames[-1])
    ax1.set_ylim(0, 1.1)
    ax1.set_ylabel('Loitering Presence')
    ax1.set_title(f'{title} - Detection Timeline')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.2)
    
    # ===== SUBPLOT 2: Loitering Scores =====
    ax2 = axes[1]
    
    # Plot loitering scores as line
    ax2.plot(frames, loitering_scores, color='purple', linewidth=2, label='Max Loitering Score')
    ax2.fill_between(frames, 0, loitering_scores, alpha=0.3, color='purple')
    
    # Highlight sustained loitering (high scores)
    for i in range(len(frames)):
        if loitering_scores[i] > 0.5:
            ax2.axvspan(frames[i]-2, frames[i]+2, alpha=0.1, color='purple')
    
    ax2.set_xlim(frames[0], frames[-1])
    ax2.set_ylim(0, 1.0)
    ax2.set_ylabel('Loitering Confidence Score')
    ax2.set_title('Loitering Detection Intensity')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.2)
    
    # ===== SUBPLOT 3: Persons Loitering =====
    ax3 = axes[2]
    
    # Bar chart of person count loitering
    colors = ['purple' if count > 0 else 'lightgray' for count in persons_loitering]
    ax3.bar(frames, persons_loitering, color=colors, alpha=0.6, width=1)
    ax3.set_xlim(frames[0], frames[-1])
    ax3.set_ylim(0, max(persons_loitering) + 1 if persons_loitering else 1)
    ax3.set_xlabel('Frame')
    ax3.set_ylabel('Number of Persons Loitering')
    ax3.set_title('Person Count - Loitering Activity')
    ax3.grid(True, alpha=0.2, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=130, bbox_inches='tight')
    plt.close()


def generate_combined_threat_heatmap(timeline, output_path, title="Threat Composition Heatmap"):
    """
    Generate a heatmap showing composition of threats (violence vs weapon vs loitering).
    
    Args:
        timeline: List of timeline entries
        output_path: Path to save the visualization
        title: Graph title
    """
    if not timeline:
        return
    
    frames = []
    violence_scores = []
    weapon_scores = []
    loitering_scores = []
    
    for entry in timeline:
        frames.append(entry['frame'])
        
        max_violence = 0.0
        max_weapon = 0.0
        max_loitering = 0.0
        
        for person in entry.get('persons', []):
            max_violence = max(max_violence, person.get('violence_score', 0.0))
            max_weapon = max(max_weapon, person.get('weapon_score', 0.0))
            max_loitering = max(max_loitering, person.get('loitering_score', 0.0))
        
        violence_scores.append(max_violence)
        weapon_scores.append(max_weapon)
        loitering_scores.append(max_loitering)
    
    # Create stacked area chart
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ax.stackplot(frames, violence_scores, weapon_scores, loitering_scores,
                 labels=['Violence', 'Weapon', 'Loitering'],
                 colors=['#ff6b6b', '#ff9500', '#9b59b6'],
                 alpha=0.7)
    
    ax.set_xlim(frames[0], frames[-1])
    ax.set_ylim(0, 1.0)
    ax.set_xlabel('Frame')
    ax.set_ylabel('Threat Component Intensity')
    ax.set_title(title)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.2)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=130, bbox_inches='tight')
    plt.close()


def generate_threat_level_distribution(timeline, output_path, title="Threat Level Distribution"):
    """
    Generate pie chart showing distribution of threat levels across frames.
    
    Args:
        timeline: List of timeline entries
        output_path: Path to save the visualization
        title: Graph title
    """
    if not timeline:
        return
    
    threat_counts = defaultdict(int)
    
    for entry in timeline:
        for person in entry.get('persons', []):
            level = person.get('threat_level', 'NORMAL')
            threat_counts[level] += 1
    
    # Sort threat levels
    order = ['NORMAL', 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
    levels = [l for l in order if l in threat_counts]
    counts = [threat_counts[l] for l in levels]
    
    # Colors for threat levels
    colors_map = {
        'NORMAL': '#2ecc71',
        'LOW': '#3498db',
        'MEDIUM': '#f39c12',
        'HIGH': '#e74c3c',
        'CRITICAL': '#c0392b'
    }
    colors = [colors_map[l] for l in levels]
    
    # Create pie chart
    fig, ax = plt.subplots(figsize=(10, 8))
    
    wedges, texts, autotexts = ax.pie(counts, labels=levels, colors=colors, 
                                       autopct='%1.1f%%', startangle=90,
                                       textprops={'fontsize': 11})
    
    # Make percentage text bold and white
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=130, bbox_inches='tight')
    plt.close()


def generate_per_person_weapon_timeline(timeline, output_path, title="Per-Person Weapon Timeline"):
    """
    Generate a timeline showing which persons had weapons detected.
    
    Args:
        timeline: List of timeline entries
        output_path: Path to save the visualization
        title: Graph title
    """
    if not timeline:
        return
    
    # Collect per-track-id weapon data
    person_weapon_data = defaultdict(list)
    frames = []
    
    for entry in timeline:
        frame_num = entry['frame']
        frames.append(frame_num)
        
        for person in entry.get('persons', []):
            track_id = person.get('track_id', -1)
            weapon_score = person.get('weapon_score', 0.0)
            person_weapon_data[track_id].append(weapon_score)
    
    if not person_weapon_data:
        return
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Prepare data for heatmap
    track_ids = sorted(person_weapon_data.keys())
    data_matrix = []
    
    for tid in track_ids:
        # Pad data to match frame count
        data = person_weapon_data[tid]
        padded = data + [0.0] * (len(frames) - len(data))
        data_matrix.append(padded[:len(frames)])
    
    data_matrix = np.array(data_matrix)
    
    # Plot heatmap
    im = ax.imshow(data_matrix, cmap='YlOrRd', aspect='auto', interpolation='nearest')
    
    # Set labels
    ax.set_yticks(range(len(track_ids)))
    ax.set_yticklabels([f'Person #{tid}' for tid in track_ids])
    ax.set_xticks(range(0, len(frames), max(1, len(frames)//10)))
    ax.set_xticklabels([str(frames[i]) for i in range(0, len(frames), max(1, len(frames)//10))])
    
    ax.set_xlabel('Frame')
    ax.set_ylabel('Tracked Person')
    ax.set_title(title)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Weapon Score', rotation=270, labelpad=15)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=130, bbox_inches='tight')
    plt.close()
