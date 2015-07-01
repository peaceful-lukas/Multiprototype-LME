figure;
hold on;


bar([lme_accuracies' lmspc_accuracies'], 'grouped');
axis([0 21 0 100]);
category_labels = {'airplane'; 'bicycle'; 'bird'; 'boat'; 'bottle'; 'bus'; 'car'; 'cat'; 'table'; 'dog';
                    'horse'; 'motorbike'; 'person'; 'plant'; 'sheep'; 'sofa'; 'train'; 'monitor'; 'cow'; 'chair'};

xticklabel_rotate([1:20],45, category_labels);


xlabel('categories');
ylabel('accuracy');
legend('LME(baseline)', 'LMSPC(ours)');
hold off;